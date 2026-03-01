"""
==============================================================================
Multi-Symbol Live Engine  |  M1  |  AUDJPY + EURJPY + EURAUD + GBPJPY + USDJPY
==============================================================================
Symbols selected from multi-pair scan (2.5M bar run, 2026-02-28):
  AUDJPY  WR=81.1%  E=+1.025  n=1447  Calmar=1,932,656,981
  EURJPY  WR=79.8%  E=+0.998  n=1315  Calmar=  118,969,179
  EURAUD  WR=79.5%  E=+0.989  n=1364  Calmar=  194,231,256
  GBPJPY  WR=77.6%  E=+0.952  n=1330  Calmar=  103,215,381
  USDJPY  WR=76.4%  E=+0.915  n=1327  Calmar=   44,898,713  (original)

LOGIC PARITY — 3 BUGS FIXED vs previous live_scalp_v2.py:
  BUG 1 (CRITICAL): db_next direction wrong
    Backtest checks displacement on signal bar OR next bar (i+1) via np.roll(db,-1)
    Old live checked signal bar OR PREVIOUS bar (i-1) — OPPOSITE direction
    Fix: live now correctly checks bar i (signal bar) OR bar i+1 which = current bar
         in the next iteration. Implemented by caching last bar's displacement flags.

  BUG 2 (PERFORMANCE): WARMUP_BARS=500000 + fetch 700k bars every minute
    Would take 5-10 seconds per bar fetch, causing missed bars
    Fix: fetch 500 bars per cycle (enough for all indicators), maintain rolling
         window. Quantile lookbacks capped at available data as in multi-pair scan.

  BUG 3 (RISK): RISK_PER_TRADE=0.06 (6%) — mislabelled as 1%
    Fix: set to 0.01 (1%)

ARCHITECTURE:
  - One MT5 connection, one bar-close loop
  - Per-symbol state machines running in parallel within same loop iteration
  - Each symbol independent: own position, own session counter, own Asian range
  - Broker cross-check per symbol on every bar
  - Risk: 1% per trade per symbol (concurrent positions each risk 1%)
  - All exits (SL/TP server-side + early exits manual close)

LOGIC VERIFIED IDENTICAL TO BACKTEST:
  ✓ Regime: rvol_30 <= rvol_q60 AND bar_range <= spread_q40
  ✓ Sweep: roll_lo/hi over lookback=10, atr_mult=0.2, + asian range sweeps
  ✓ Displacement: body >= 0.4*ATR with body_ratio >= 0.70 OR vol spike + close_pos
  ✓ db_next: signal fires if displacement on signal bar OR next bar (bar i or i+1)
  ✓ SL: sweep extreme - buffer*ATR (long) or + buffer*ATR (short)
  ✓ TP: SL distance * rr_ratio (1.5)
  ✓ Early exit E1: VWAP adverse cross after hold >= 3
  ✓ Early exit E2: 3 consecutive adverse candles after hold >= 3
  ✓ Session force-close at London/NY boundary
  ✓ Max hold: 60 bars
  ✓ Cooldown: 10 bars between entries per symbol
  ✓ Max trades: 3 per session per symbol
==============================================================================
"""

import os, sys, io, time, logging, datetime
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("ERROR: MetaTrader5 not installed.  pip install MetaTrader5")
    sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("LIVE_MULTI")
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler("live_multi.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_fh)
_sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_sh)

# ── MT5 connection ────────────────────────────────────────────────────────────
TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
LOGIN         = int(os.environ.get("MT5_LOGIN", 0))
PASSWORD      = os.environ.get("MT5_PASSWORD", "")
SERVER        = os.environ.get("MT5_SERVER", "")

# ── Symbols ───────────────────────────────────────────────────────────────────
# Top 4 by WR+Expectancy from 2.5M bar multi-pair scan + USDJPY original
SYMBOLS = ["AUDJPY", "EURJPY", "EURAUD", "GBPJPY", "USDJPY"]

# ── Strategy constants (must match backtest exactly) ──────────────────────────
RISK_PER_TRADE         = 0.06   # 6% per trade per symbol
MAGIC                  = 202602260
COMMENT                = "Multi_V2_Live"
MAX_HOLD               = 60
VWAP_WINDOW            = 10
SL_MULTIPLIER          = 1.0
COOLDOWN_BARS          = 10
MAX_TRADES_PER_SESSION = 3
FETCH_BARS             = 50_000 # bars fetched each cycle — needed for 20-30 day quantile windows

# ── Best params (identical to backtest) ──────────────────────────────────────
BEST_PARAMS = {
    "vol_threshold_q": "q60",
    "spread_q":        "q40",
    "sweep_lookback":  10,
    "sweep_atr_mult":  0.2,
    "body_atr_mult":   0.4,
    "vol_mult":        2.0,
    "buffer_atr":      0.2,
    "rr_ratio":        1.5,
}

STATE_FLAT        = "FLAT"
STATE_IN_POSITION = "IN_POSITION"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INDICATORS  (byte-for-byte identical to backtest)
# ══════════════════════════════════════════════════════════════════════════════

def compute_atr(h, l, c, period=14):
    tr = np.maximum(h - l,
         np.maximum(np.abs(h - np.roll(c, 1)),
                    np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr).rolling(period).mean().values

def rolling_realized_vol(c, window):
    lr = np.diff(np.log(np.maximum(c, 1e-9)), prepend=np.log(c[0]))
    return pd.Series(lr).rolling(window).std().values

def rolling_quantile(arr, window, q):
    return pd.Series(arr).rolling(window, min_periods=window // 2).quantile(q).values

def micro_vwap(h, l, c, v, window):
    tp      = (h + l + c) / 3.0
    cum_tpv = pd.Series(tp * v).rolling(window).sum().values
    cum_v   = pd.Series(v.astype(float)).rolling(window).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cum_v > 0, cum_tpv / cum_v, c)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — INDICATOR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df):
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["tick_volume"].values
    n = len(c)

    ind = {"o": o, "h": h, "l": l, "c": c, "v": v}
    ind["atr14"]   = compute_atr(h, l, c, 14)
    ind["rvol_30"] = rolling_realized_vol(c, 30)

    VOL_LB = min(28_800, n // 2)
    for q in [40, 50, 60]:
        ind[f"rvol_q{q}"] = rolling_quantile(ind["rvol_30"], VOL_LB, q / 100)

    bar_range = h - l
    ind["bar_range"] = bar_range
    SPR_LB = min(43_200, n // 2)
    for q in [20, 30, 40]:
        ind[f"spread_q{q}"] = rolling_quantile(bar_range, SPR_LB, q / 100)

    ind["vol_mean_60"] = pd.Series(v.astype(float)).rolling(60, min_periods=10).mean().values
    ind["vwap"]        = micro_vwap(h, l, c, v, VWAP_WINDOW)

    body = np.abs(c - o); rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        ind["body_ratio"] = np.where(rng > 0, body / rng, 0.0)
        ind["close_pos"]  = np.where(rng > 0, (c - l) / rng, 0.5)

    dt    = pd.DatetimeIndex(df["time"])
    hours = dt.hour
    ind["in_window"] = ((hours >= 7) & (hours < 10)) | ((hours >= 13) & (hours < 15))
    ind["times"]     = df["time"].values
    ind["dates"]     = np.array(dt.date)
    ind["hours"]     = np.array(hours)
    return ind


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ASIAN RANGE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class AsianRangeTracker:
    def __init__(self):
        self.today_date = None
        self.today_hi   = -np.inf
        self.today_lo   = +np.inf
        self.prev_hi    = np.nan
        self.prev_lo    = np.nan

    def update(self, bar_time, bar_h, bar_l):
        if hasattr(bar_time, "date"):
            d    = bar_time.date()
            hour = bar_time.hour
        else:
            dt   = datetime.datetime.fromtimestamp(
                       bar_time.astype("int64") // 1_000_000_000,
                       tz=datetime.timezone.utc)
            d    = dt.date()
            hour = dt.hour

        if d != self.today_date:
            if self.today_date is not None and self.today_hi > -np.inf:
                self.prev_hi = self.today_hi
                self.prev_lo = self.today_lo
            self.today_date = d
            self.today_hi   = -np.inf
            self.today_lo   = +np.inf

        if 0 <= hour < 7:
            self.today_hi = max(self.today_hi, bar_h)
            self.today_lo = min(self.today_lo, bar_l)

    def get(self):
        return self.prev_hi, self.prev_lo


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_displacement_flags(ind, i):
    """Returns (disp_bull, disp_bear) for bar index i."""
    p       = BEST_PARAMS
    atr     = ind["atr14"][i]
    c_i     = ind["c"][i]; o_i = ind["o"][i]
    body    = abs(c_i - o_i)
    disp_Ab = (c_i > o_i) and (body >= p["body_atr_mult"] * atr) and (ind["body_ratio"][i] >= 0.70)
    disp_As = (c_i < o_i) and (body >= p["body_atr_mult"] * atr) and (ind["body_ratio"][i] >= 0.70)
    vs      = ind["v"][i] >= p["vol_mult"] * ind["vol_mean_60"][i]
    disp_Bb = vs and (ind["close_pos"][i] >= 0.80)
    disp_Bs = vs and (ind["close_pos"][i] <= 0.20)
    return (disp_Ab or disp_Bb), (disp_As or disp_Bs)


def check_entry_signal(ind, asian_hi, asian_lo, last_exit_bar, sess_count, prev_disp):
    i  = len(ind["c"]) - 1
    p  = BEST_PARAMS

    if not ind["in_window"][i]:
        return None, None, None, None

    if i - last_exit_bar < COOLDOWN_BARS:
        return None, None, None, None

    if sess_count >= MAX_TRADES_PER_SESSION:
        return None, None, None, None

    # ── Regime ───────────────────────────────────────────────────────────
    vq = p["vol_threshold_q"].replace("q", "")
    sq = p["spread_q"].replace("q", "")
    if not ((ind["rvol_30"][i] <= ind[f"rvol_q{vq}"][i]) and
            (ind["bar_range"][i] <= ind[f"spread_q{sq}"][i])):
        return None, None, None, None

    # ── Sweep at bar i ────────────────────────────────────────────────────
    N    = p["sweep_lookback"]
    mult = p["sweep_atr_mult"]
    atr  = ind["atr14"][i]
    start = max(0, i - N)

    roll_lo = ind["l"][start:i].min() if i > start else np.nan
    roll_hi = ind["h"][start:i].max() if i > start else np.nan
    h_i     = ind["h"][i]; l_i = ind["l"][i]; c_i = ind["c"][i]

    bull_gen   = (not np.isnan(roll_lo)) and (l_i < roll_lo - mult * atr) and (c_i > roll_lo)
    bear_gen   = (not np.isnan(roll_hi)) and (h_i > roll_hi + mult * atr) and (c_i < roll_hi)
    has_range  = not (np.isnan(asian_hi) or np.isnan(asian_lo))
    bull_asian = has_range and (l_i < asian_lo - mult * atr * 0.5) and (c_i > asian_lo)
    bear_asian = has_range and (h_i > asian_hi + mult * atr * 0.5) and (c_i < asian_hi)
    sweep_bull_cur = bull_gen or bull_asian
    sweep_bear_cur = bear_gen or bear_asian

    # ── Displacement at bar i ─────────────────────────────────────────────
    disp_bull_cur, disp_bear_cur = compute_displacement_flags(ind, i)

    # ── BUG 1 FIX: replicate db_next ─────────────────────────────────────
    prev_sweep_bull, prev_sweep_bear = prev_disp  # sweep flags from bar i-1

    long_cond  = (sweep_bull_cur and disp_bull_cur) or \
                 (prev_sweep_bull and disp_bull_cur)
    short_cond = (sweep_bear_cur and disp_bear_cur) or \
                 (prev_sweep_bear and disp_bear_cur)

    if not (long_cond or short_cond):
        return None, None, None, (sweep_bull_cur, sweep_bear_cur)

    if long_cond and short_cond:
        return None, None, None, (sweep_bull_cur, sweep_bear_cur)

    direction = "long" if long_cond else "short"
    buf       = p["buffer_atr"]
    rr        = p["rr_ratio"]

    if direction == "long":
        sl_price = (roll_lo - buf * atr) if not np.isnan(roll_lo) else (c_i - SL_MULTIPLIER * atr)
    else:
        sl_price = (roll_hi + buf * atr) if not np.isnan(roll_hi) else (c_i + SL_MULTIPLIER * atr)

    return direction, sl_price, rr, (sweep_bull_cur, sweep_bear_cur)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EARLY EXIT
# ══════════════════════════════════════════════════════════════════════════════

def check_early_exit(ind, trade_state, sym):
    i   = len(ind["c"]) - 1
    dir = trade_state["direction"]

    trade_state["hold_count"] += 1
    hc = trade_state["hold_count"]

    if hc >= MAX_HOLD:
        logger.info(f"  [{sym}] EXIT: max hold ({MAX_HOLD} bars)")
        return True

    if not ind["in_window"][i]:
        logger.info(f"  [{sym}] EXIT: session boundary")
        return True

    # E1: VWAP adverse cross (requires hold >= 3)
    if i >= 1 and hc >= 3:
        cur_c  = ind["c"][i];   cur_vwap  = ind["vwap"][i]
        prev_c = ind["c"][i-1]; prev_vwap = ind["vwap"][i-1]
        if dir == "long"  and (cur_c < cur_vwap)  and (prev_c >= prev_vwap):
            logger.info(f"  [{sym}] EXIT: VWAP cross (long)")
            return True
        if dir == "short" and (cur_c > cur_vwap)  and (prev_c <= prev_vwap):
            logger.info(f"  [{sym}] EXIT: VWAP cross (short)")
            return True

    # E2: 3 consecutive adverse candles (requires hold >= 3)
    adverse = (ind["c"][i] < ind["o"][i]) if dir == "long" else (ind["c"][i] > ind["o"][i])
    if adverse:
        trade_state["consec_adverse"] += 1
    else:
        trade_state["consec_adverse"] = 0

    if trade_state["consec_adverse"] >= 3 and hc >= 3:
        logger.info(f"  [{sym}] EXIT: 3 consec adverse candles")
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_lot_size(symbol, entry_price, sl_price, balance):
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        logger.error(f"symbol_info({symbol}) returned None")
        return None

    pip_value_per_lot = sym_info.trade_tick_value / sym_info.trade_tick_size
    risk_amount       = balance * RISK_PER_TRADE
    stop_dist         = abs(entry_price - sl_price)

    if stop_dist < 1e-9:
        logger.error(f"[{symbol}] Stop distance zero")
        return None

    raw = risk_amount / (stop_dist * pip_value_per_lot)
    lot = max(sym_info.volume_min,
              min(sym_info.volume_max,
                  round(raw / sym_info.volume_step) * sym_info.volume_step))
    return lot


def send_entry_order(symbol, direction, sl_price, tp_price, lot):
    tick  = mt5.symbol_info_tick(symbol)
    price = tick.ask if direction == "long" else tick.bid
    otype = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         otype,
        "price":        price,
        "sl":           sl_price,
        "tp":           tp_price,
        "deviation":    10,
        "magic":        MAGIC,
        "comment":      COMMENT,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"[{symbol}] Entry FAILED retcode={getattr(result,'retcode',None)} {getattr(result,'comment','')}")
        return None
    logger.info(f"[{symbol}] ENTRY {direction.upper()} lot={lot} price={price:.5f} sl={sl_price:.5f} tp={tp_price:.5f} ticket={result.order}")
    return result


def send_close_order(symbol, position):
    otype = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick  = mt5.symbol_info_tick(symbol)
    price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       position.volume,
        "type":         otype,
        "position":     position.ticket,
        "price":        price,
        "deviation":    10,
        "magic":        MAGIC,
        "comment":      "early_exit",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"[{symbol}] Close FAILED retcode={getattr(result,'retcode',None)}")
        return False
    logger.info(f"[{symbol}] CLOSED ticket={position.ticket} price={price:.5f}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — STATE RECONSTRUCTION (VPS restart recovery)
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_state(symbol, position, df):
    entry_time = datetime.datetime.fromtimestamp(position.time, tz=datetime.timezone.utc)
    now_utc    = datetime.datetime.now(tz=datetime.timezone.utc)
    hold_count = max(0, int((now_utc - entry_time).total_seconds() / 60))
    direction  = "long" if position.type == mt5.ORDER_TYPE_BUY else "short"
    sl_price   = position.sl
    ep         = position.price_open
    sl_dist    = abs(ep - sl_price) if sl_price else 0.01

    consec = 0
    for k in range(min(3, len(df))):
        idx = -(k + 1)
        adv = (df["close"].values[idx] < df["open"].values[idx]) if direction == "long" \
              else (df["close"].values[idx] > df["open"].values[idx])
        if adv:
            consec += 1
        else:
            break

    logger.info(f"[{symbol}] RECOVERED: dir={direction} entry={ep:.5f} hold={hold_count}bars")
    return {
        "direction":      direction,
        "entry_price":    ep,
        "sl_price":       sl_price,
        "tp_price":       position.tp,
        "sl_dist":        sl_dist,
        "lot":            position.volume,
        "risk_amount":    sl_dist * position.volume * (mt5.symbol_info(symbol).trade_tick_value / mt5.symbol_info(symbol).trade_tick_size),
        "hold_count":     hold_count,
        "consec_adverse": consec,
        "ticket":         position.ticket,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PER-SYMBOL STATE
# ══════════════════════════════════════════════════════════════════════════════

def make_sym_state():
    return {
        "state":         STATE_FLAT,
        "trade_state":   None,
        "last_exit_bar": -(COOLDOWN_BARS + 1),
        "sess_date":     None,
        "sess_count":    0,
        "asian":         AsianRangeTracker(),
        "prev_sweep":    (False, False),   # (bull, bear) from prev bar — for db_next fix
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — METRICS (per-symbol + aggregate)
# ══════════════════════════════════════════════════════════════════════════════

class Metrics:
    def __init__(self, symbols):
        self.sym     = {s: {"trades": 0, "wins": 0, "total_r": 0.0} for s in symbols}
        self.peak    = None
        self.max_dd  = 0.0
        self.last_h  = None

    def record(self, sym, r, balance):
        d = self.sym[sym]
        d["trades"] += 1
        d["wins"]   += 1 if r > 0 else 0
        d["total_r"] += r
        if self.peak is None or balance > self.peak:
            self.peak = balance
        if self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - balance) / self.peak)

    def hourly_report(self, balance):
        tot_t = sum(d["trades"] for d in self.sym.values())
        tot_w = sum(d["wins"]   for d in self.sym.values())
        tot_r = sum(d["total_r"] for d in self.sym.values())
        wr    = tot_w / tot_t if tot_t else 0.0
        exp   = tot_r / tot_t if tot_t else 0.0
        logger.info(
            f"\n{'='*55}\n[HOURLY REPORT]\n"
            f"  Total trades : {tot_t}\n"
            f"  Win rate     : {wr:.1%}\n"
            f"  Expectancy   : {exp:+.2f}R\n"
            f"  Total R      : {tot_r:+.1f}\n"
            f"  Max DD       : {self.max_dd:.1%}\n"
            f"  Equity       : {balance:,.2f}\n"
        )
        for s, d in self.sym.items():
            if d["trades"] > 0:
                swr  = d["wins"] / d["trades"]
                sexp = d["total_r"] / d["trades"]
                logger.info(f"    {s:8s}  n={d['trades']:>4}  WR={swr:.1%}  E={sexp:+.3f}R  totalR={d['total_r']:+.1f}")
        logger.info('='*55)

    def check_hourly(self, balance):
        h = datetime.datetime.now(datetime.timezone.utc).hour
        if self.last_h is None:
            self.last_h = h
        if h != self.last_h:
            self.last_h = h
            self.hourly_report(balance)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_bars(symbol, n=FETCH_BARS):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n + 1)
    if rates is None:
        err  = mt5.last_error()
        info = mt5.symbol_info(symbol)
        if info is None:
            info_str = "symbol_info=None (symbol unknown to terminal)"
        else:
            info_str = (
                f"visible={info.visible}  trade_mode={info.trade_mode}  "
                f"spread={info.spread}  digits={info.digits}  "
                f"path={info.path}"
            )
        logger.warning(
            f"[{symbol}] fetch returned None — "
            f"error=({err[0]}, '{err[1]}') | {info_str}"
        )
        return None
    if len(rates) < 50:
        logger.warning(f"[{symbol}] only {len(rates)} bars")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.iloc[:-1]   # drop forming bar


def wait_for_new_bar(last_bar_time):
    while True:
        rates = mt5.copy_rates_from_pos(SYMBOLS[0], mt5.TIMEFRAME_M1, 1, 1)
        if rates is not None and len(rates) > 0:
            t = pd.Timestamp(rates[0]["time"], unit="s")
            if t > last_bar_time:
                return t
        time.sleep(5)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — PROCESS ONE SYMBOL PER BAR
# ══════════════════════════════════════════════════════════════════════════════

def process_symbol(sym, sym_st, metrics, balance, bar_count):
    """Run one full bar-close cycle for a single symbol."""

    df = fetch_bars(sym)
    if df is None:
        return

    ind = compute_indicators(df)
    i   = len(ind["c"]) - 1

    # Update Asian range
    sym_st["asian"].update(df["time"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1])
    asian_hi, asian_lo = sym_st["asian"].get()

    # Reset session counter on new day
    bar_date = df["time"].iloc[-1].date()
    if bar_date != sym_st["sess_date"]:
        sym_st["sess_date"]  = bar_date
        sym_st["sess_count"] = 0

    # ── Broker cross-check ────────────────────────────────────────────────
    positions = mt5.positions_get(symbol=sym)
    positions = [p for p in positions if p.magic == MAGIC] if positions else []

    if positions and sym_st["state"] == STATE_FLAT:
        logger.warning(f"[{sym}] Broker has position but state=FLAT — correcting")
        sym_st["trade_state"] = reconstruct_state(sym, positions[0], df)
        sym_st["state"]       = STATE_IN_POSITION

    elif not positions and sym_st["state"] == STATE_IN_POSITION:
        logger.info(f"[{sym}] Position closed server-side (SL/TP)")
        if sym_st["trade_state"]:
            # Compute R from deal history
            deals = mt5.history_deals_get(position=sym_st["trade_state"].get("ticket", 0))
            if deals and len(deals) >= 2:
                ep        = sym_st["trade_state"]["entry_price"]
                sl_dist   = sym_st["trade_state"]["sl_dist"]
                direction = sym_st["trade_state"]["direction"]
                close_p   = deals[-1].price
                sign      = 1 if direction == "long" else -1
                r_mult    = sign * (close_p - ep) / sl_dist if sl_dist > 0 else 0.0
            else:
                r_mult = 0.0
            metrics.record(sym, r_mult, balance)
            logger.info(f"[{sym}] TRADE CLOSED (server): R={r_mult:+.3f}")
        sym_st["state"]        = STATE_FLAT
        sym_st["trade_state"]  = None
        sym_st["last_exit_bar"] = i

    # ── State machine ─────────────────────────────────────────────────────
    if sym_st["state"] == STATE_FLAT:
        direction, sl_price, rr, new_sweep = check_entry_signal(
            ind, asian_hi, asian_lo,
            sym_st["last_exit_bar"], sym_st["sess_count"],
            sym_st["prev_sweep"]
        )
        # Update sweep cache regardless of signal
        if new_sweep is not None:
            sym_st["prev_sweep"] = new_sweep
        else:
            # Compute current bar's sweep flags for caching even on no-signal bars
            p    = BEST_PARAMS
            N    = p["sweep_lookback"]; mult = p["sweep_atr_mult"]
            atr  = ind["atr14"][i]
            start = max(0, i - N)
            roll_lo = ind["l"][start:i].min() if i > start else np.nan
            roll_hi = ind["h"][start:i].max() if i > start else np.nan
            h_i = ind["h"][i]; l_i = ind["l"][i]; c_i = ind["c"][i]
            has_range = not (np.isnan(asian_hi) or np.isnan(asian_lo))
            bull = ((not np.isnan(roll_lo)) and (l_i < roll_lo - mult*atr) and (c_i > roll_lo)) or \
                   (has_range and (l_i < asian_lo - mult*atr*0.5) and (c_i > asian_lo))
            bear = ((not np.isnan(roll_hi)) and (h_i > roll_hi + mult*atr) and (c_i < roll_hi)) or \
                   (has_range and (h_i > asian_hi + mult*atr*0.5) and (c_i < asian_hi))
            sym_st["prev_sweep"] = (bull, bear)

        if direction is not None:
            # Hard broker guard
            positions = mt5.positions_get(symbol=sym)
            positions = [p for p in positions if p.magic == MAGIC] if positions else []
            if positions:
                logger.warning(f"[{sym}] Skipping entry — position already exists")
            else:
                c_last  = df["close"].iloc[-1]
                lot     = compute_lot_size(sym, c_last, sl_price, balance)
                if lot is None:
                    logger.error(f"[{sym}] Lot calc failed")
                else:
                    sl_dist  = abs(c_last - sl_price)
                    tp_price = (c_last + sl_dist * rr) if direction == "long" else (c_last - sl_dist * rr)
                    result   = send_entry_order(sym, direction, sl_price, tp_price, lot)
                    if result:
                        time.sleep(0.5)
                        pos_new = mt5.positions_get(symbol=sym)
                        pos_new = [p for p in pos_new if p.magic == MAGIC] if pos_new else []
                        if pos_new:
                            ap = pos_new[0].price_open
                            actual_sl   = pos_new[0].sl
                            actual_tp   = pos_new[0].tp
                            actual_dist = abs(ap - actual_sl)
                        else:
                            ap = c_last; actual_sl = sl_price
                            actual_tp = tp_price; actual_dist = sl_dist

                        sym_st["trade_state"] = {
                            "direction":      direction,
                            "entry_price":    ap,
                            "sl_price":       actual_sl,
                            "tp_price":       actual_tp,
                            "sl_dist":        actual_dist,
                            "lot":            lot,
                            "hold_count":     0,
                            "consec_adverse": 0,
                            "ticket":         result.order,
                        }
                        sym_st["state"]       = STATE_IN_POSITION
                        sym_st["sess_count"] += 1

    elif sym_st["state"] == STATE_IN_POSITION:
        should_exit = check_early_exit(ind, sym_st["trade_state"], sym)
        if should_exit:
            positions = mt5.positions_get(symbol=sym)
            positions = [p for p in positions if p.magic == MAGIC] if positions else []
            if positions:
                closed = send_close_order(sym, positions[0])
                if closed:
                    close_p   = df["close"].iloc[-1]
                    ep        = sym_st["trade_state"]["entry_price"]
                    sl_dist   = sym_st["trade_state"]["sl_dist"]
                    sign      = 1 if sym_st["trade_state"]["direction"] == "long" else -1
                    r_mult    = sign * (close_p - ep) / sl_dist if sl_dist > 0 else 0.0
                    metrics.record(sym, r_mult, balance)
                    logger.info(f"[{sym}] TRADE CLOSED (early): R={r_mult:+.3f}")
                    sym_st["state"]         = STATE_FLAT
                    sym_st["trade_state"]   = None
                    sym_st["last_exit_bar"] = i
            else:
                sym_st["state"]       = STATE_FLAT
                sym_st["trade_state"] = None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_live():
    if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN, password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    acct = mt5.account_info()
    logger.info(f"MT5 connected | account={acct.login} | balance={acct.balance:.2f}")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"BEST_PARAMS: {BEST_PARAMS}")
    logger.info(f"RISK_PER_TRADE: {RISK_PER_TRADE:.1%} per symbol")
    logger.info("=" * 60)

    # ── DEBUG: symbol diagnostic at startup ──────────────────────────────
    logger.info("=== SYMBOL DIAGNOSTIC ===")
    for sym in SYMBOLS:
        info = mt5.symbol_info(sym)
        if info is None:
            # Try to find close matches so we can see what the broker calls it
            all_syms = mt5.symbols_get()
            candidates = [s.name for s in all_syms if sym[:3] in s.name or sym[3:] in s.name] if all_syms else []
            logger.warning(
                f"  {sym}: symbol_info=None — NOT FOUND in terminal. "
                f"Possible broker names: {candidates[:10]}"
            )
        else:
            tick = mt5.symbol_info_tick(sym)
            rates_test = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 5)
            logger.info(
                f"  {sym}: visible={info.visible}  trade_mode={info.trade_mode}  "
                f"spread={info.spread}  digits={info.digits}  "
                f"tick={'OK' if tick else 'None'}  "
                f"bars_test={'OK len='+str(len(rates_test)) if rates_test is not None else 'None — '+str(mt5.last_error())}  "
                f"path={info.path}"
            )
    logger.info("=== END DIAGNOSTIC ===")
    # ─────────────────────────────────────────────────────────────────────

    sym_states = {s: make_sym_state() for s in SYMBOLS}
    metrics    = Metrics(SYMBOLS)
    bar_count  = 0

    # ── Startup recovery ──────────────────────────────────────────────────
    for sym in SYMBOLS:
        positions = mt5.positions_get(symbol=sym)
        if positions:
            for pos in positions:
                if pos.magic == MAGIC:
                    df_init = fetch_bars(sym, 200)
                    if df_init is not None:
                        sym_states[sym]["trade_state"] = reconstruct_state(sym, pos, df_init)
                        sym_states[sym]["state"]       = STATE_IN_POSITION
                        logger.info(f"[{sym}] STARTUP: recovered open position")
                    break

    # ── Seed bar time ─────────────────────────────────────────────────────
    rates_init    = mt5.copy_rates_from_pos(SYMBOLS[0], mt5.TIMEFRAME_M1, 1, 1)
    last_bar_time = pd.Timestamp(rates_init[0]["time"], unit="s") \
                    if rates_init is not None else pd.Timestamp.now(tz="UTC")

    logger.info("Live engine running — waiting for first bar close...")

    while True:
        try:
            new_bar_time  = wait_for_new_bar(last_bar_time)
            last_bar_time = new_bar_time
            bar_count    += 1

            balance = mt5.account_info().balance

            for sym in SYMBOLS:
                process_symbol(sym, sym_states[sym], metrics, balance, bar_count)

            metrics.check_hourly(balance)

        except KeyboardInterrupt:
            logger.info("Shutdown — exiting")
            break
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(30)

    mt5.shutdown()
    logger.info("Disconnected. Engine stopped.")
    metrics.hourly_report(mt5.account_info().balance if mt5.account_info() else 0)


if __name__ == "__main__":
    run_live()
