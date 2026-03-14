"""
==============================================================================
OIF  —  LIVE EXECUTION ENGINE  |  M5
==============================================================================

PARITY AUDIT vs BACKTEST (oif_portfolio2.py):
  ✓ Session mask     : both = UTC 07-11 OR 13-17
  ✓ ATR threshold    : atr14[i] > shift(1).rolling(50).median() * 1.1
                       (bar i NOT in its own threshold — lookahead fix)
  ✓ RV percentile    : expanding window, bisect_right before insort
                       rv50_pct[i] >= 30 (all symbols)
  ✓ EMA overext      : |c[i] - ema50[i-1]| > ext_atr_mult * atr14[i]
                       (ema50 shifted by 1 — lookahead fix)
  ✓ Swing hi/lo      : shift(1).rolling(5).max/min
                       (bar i NOT in swing window — lookahead fix)
  ✓ Big body         : abs(c-o) > body_atr_mult * atr14
  ✓ Direction        : bear candle → long fade | bull candle → short fade
  ✓ Cooldown         : shared counter long+short, per symbol
  ✓ Signal bar       : bar i (closed) — entry bar i+1 (next open)
  ✓ SL               : low[i]  - sl_atr_mult * atr14[i]  (long)
                       high[i] + sl_atr_mult * atr14[i]  (short)
  ✓ TP               : (open[i] + close[i]) / 2  (imbalance midpoint)
  ✓ min_rr gate      : tp_dist / sl_dist >= min_rr
  ✓ direction_ok gate: tp above entry (long) / tp below entry (short)
  ✓ MAX_HOLD         : 48 bars (240 min) — force close

CONNECTION: copied from Septillion (DO NOT CHANGE):
  - MT5 env vars, IOC filling, deviation=10
  - Bar clock on M5 reference symbol
  - compute_lot_size: risk_amount / stop_dist / pip_value
  - Broker cross-check: positions_get filtered by magic+comment
  - reconstruct_state on restart

MAGIC  : 202603050
COMMENT: "OIF"
LOG    : oif_live.log
==============================================================================
"""

import os, sys, io, time, logging, datetime, bisect
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
logger = logging.getLogger("OIF_LIVE")
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler("oif_live.log", maxBytes=10_000_000, backupCount=5,
                           encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_fh)
_sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer,
                             encoding="utf-8", errors="replace"))
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_sh)

# ── MT5 connection (from Septillion) ─────────────────────────────────────────
TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH",
                               r"C:\Program Files\MetaTrader 5\terminal64.exe")
LOGIN    = int(os.environ.get("MT5_LOGIN",    0))
PASSWORD =     os.environ.get("MT5_PASSWORD", "")
SERVER   =     os.environ.get("MT5_SERVER",   "")

# ── Engine identity ───────────────────────────────────────────────────────────
MAGIC   = 202603050   # unique — Septillion=202603040
COMMENT = "OIF"

# ── Strategy constants (must match backtest exactly) ──────────────────────────
RISK_PER_TRADE  = 0.01    # 1% per trade per symbol
MAX_HOLD        = 48      # 48 M5 bars = 240 min
ATR_PERIOD      = 14
ATR_LB          = 50      # rolling median lookback
ATR_FILTER_MULT = 1.1
RV_LB           = 50      # realised vol lookback
EMA_PERIOD      = 50
SWING_LB        = 5       # swing high/low lookback
WARMUP          = 80      # bars needed before any signal is valid
FETCH_BARS      = 3000    # M5 bars to fetch per symbol (≈10 days, enough for all indicators)

# ── Locked params per symbol (from grid search) ───────────────────────────────
SYMBOL_PARAMS = {
    "AUDUSD": {"sl_atr_mult": 0.30, "body_atr_mult": 1.20, "ext_atr_mult": 0.50,
               "min_rr": 0.60, "rv_pct_thresh": 30, "cooldown_bars": 6},
    "EURGBP": {"sl_atr_mult": 0.30, "body_atr_mult": 1.20, "ext_atr_mult": 0.50,
               "min_rr": 0.80, "rv_pct_thresh": 30, "cooldown_bars": 3},
    "EURUSD": {"sl_atr_mult": 0.30, "body_atr_mult": 1.20, "ext_atr_mult": 0.50,
               "min_rr": 0.80, "rv_pct_thresh": 30, "cooldown_bars": 3},
    "GBPUSD": {"sl_atr_mult": 0.30, "body_atr_mult": 1.20, "ext_atr_mult": 0.50,
               "min_rr": 0.80, "rv_pct_thresh": 30, "cooldown_bars": 3},
    "USDJPY": {"sl_atr_mult": 0.30, "body_atr_mult": 1.20, "ext_atr_mult": 0.50,
               "min_rr": 0.80, "rv_pct_thresh": 30, "cooldown_bars": 3},
}

# Session: both london + ny (UTC) — same as backtest "both"
SESSION_HOURS_UTC = lambda h: ((h >= 7) and (h < 11)) or ((h >= 13) and (h < 17))

PAIRS = list(SYMBOL_PARAMS.keys())
STATE_FLAT        = "FLAT"
STATE_IN_POSITION = "IN_POSITION"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INDICATORS
#  All computed on the FULL bar array each tick.
#  No incremental tricks — simplicity = parity with backtest.
#  WARMUP guard ensures no signal fires until all indicators are valid.
# ══════════════════════════════════════════════════════════════════════════════

def _atr_wilder(h, l, c, period=14):
    n  = len(h)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(h[1:] - l[1:],
             np.maximum(np.abs(h[1:] - c[:-1]),
                        np.abs(l[1:] - c[:-1])))
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = float(tr[:period].mean())
    k = 1.0 / period
    for i in range(period, n):
        out[i] = out[i-1] * (1.0 - k) + tr[i] * k
    return out


def _ema(arr, period):
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    out[period - 1] = float(np.nanmean(arr[:period]))
    k = 2.0 / (period + 1)
    for i in range(period, len(arr)):
        out[i] = out[i-1] * (1.0 - k) + arr[i] * k
    return out


def _rv50(c):
    """50-bar realised vol — matches backtest exactly."""
    log_ret = np.empty(len(c))
    log_ret[0] = np.nan
    log_ret[1:] = np.log(c[1:] / c[:-1])
    return (pd.Series(log_ret)
            .rolling(RV_LB, min_periods=RV_LB)
            .std()
            .values * np.sqrt(RV_LB))


def _rv_expanding_pct(rv_arr, warmup):
    """
    Expanding percentile rank — identical to backtest _expanding_percentile_rank.
    bisect_right BEFORE insort so current bar is ranked against prior values only.
    """
    n   = len(rv_arr)
    out = np.full(n, np.nan)
    hist = []
    for i in range(warmup, n):
        v = rv_arr[i]
        if np.isnan(v):
            continue
        pos = bisect.bisect_right(hist, v)
        bisect.insort(hist, v)
        out[i] = (pos / len(hist)) * 100.0
    return out


def build_indicators(df):
    """
    Compute all indicators from a full DataFrame of closed M5 bars.
    Returns dict of arrays. Last element = most recent closed bar.
    All fixed parity with backtest.
    """
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)

    atr14 = _atr_wilder(h, l, c, ATR_PERIOD)

    # ATR threshold: shift(1).rolling(50).median() * 1.1
    # shift(1) = bar i NOT included in its own threshold (parity fix)
    atr_thr = (pd.Series(atr14)
               .shift(1)
               .rolling(ATR_LB, min_periods=ATR_LB // 2)
               .median()
               .values * ATR_FILTER_MULT)

    # Swing high/low: shift(1).rolling(5) — bar i NOT in window (parity fix)
    swing_h = pd.Series(h).shift(1).rolling(SWING_LB, min_periods=1).max().values
    swing_l = pd.Series(l).shift(1).rolling(SWING_LB, min_periods=1).min().values

    # EMA50, shifted by 1 for overext calc (parity fix)
    ema50      = _ema(c, EMA_PERIOD)
    ema50_prev = pd.Series(ema50).shift(1).values

    # Realised vol + expanding percentile
    rv50     = _rv50(c)
    rv50_pct = _rv_expanding_pct(rv50, WARMUP)

    body       = np.abs(c - o)
    candle_rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = np.where(candle_rng > 1e-10, body / candle_rng, np.nan)

    # UTC hour for session filter
    utc_hours = df["time_utc"].dt.hour.values

    return {
        "o": o, "h": h, "l": l, "c": c,
        "atr14":     atr14,
        "atr_thr":   atr_thr,
        "swing_h":   swing_h,
        "swing_l":   swing_l,
        "ema50_prev":ema50_prev,
        "rv50_pct":  rv50_pct,
        "body":      body,
        "body_ratio":body_ratio,
        "utc_hours": utc_hours,
        "n":         len(c),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SIGNAL CHECK ON LAST CLOSED BAR
#  Evaluates bar i = len-1 (the bar that just closed).
#  Entry will be on bar i+1 (next open = current market price).
# ══════════════════════════════════════════════════════════════════════════════

def check_signal(ind, params, bars_since_last_signal):
    """
    Returns: ("long" | "short" | None, sl_price, tp_price, eff_rr)
    Checks ONLY the last closed bar against backtest conditions.
    """
    i = ind["n"] - 1

    # ── Guard: warmup ─────────────────────────────────────────────────────────
    if i < WARMUP:
        return None, None, None, None

    # ── Cooldown ──────────────────────────────────────────────────────────────
    if bars_since_last_signal < params["cooldown_bars"]:
        return None, None, None, None

    # ── Session filter (UTC) ──────────────────────────────────────────────────
    h_utc = int(ind["utc_hours"][i])
    if not SESSION_HOURS_UTC(h_utc):
        return None, None, None, None

    # ── ATR regime ────────────────────────────────────────────────────────────
    atr  = ind["atr14"][i]
    athr = ind["atr_thr"][i]
    if np.isnan(atr) or np.isnan(athr) or atr <= athr:
        return None, None, None, None

    # ── RV percentile ─────────────────────────────────────────────────────────
    rvp = ind["rv50_pct"][i]
    if np.isnan(rvp) or rvp < params["rv_pct_thresh"]:
        return None, None, None, None

    # ── Big body ──────────────────────────────────────────────────────────────
    body = ind["body"][i]
    if body <= params["body_atr_mult"] * atr:
        return None, None, None, None

    # ── Overextension from EMA50[i-1] ────────────────────────────────────────
    ema_p = ind["ema50_prev"][i]
    if np.isnan(ema_p):
        return None, None, None, None
    c_i = ind["c"][i]
    if abs(c_i - ema_p) <= params["ext_atr_mult"] * atr:
        return None, None, None, None

    # ── Structure break ───────────────────────────────────────────────────────
    sh = ind["swing_h"][i]
    sl = ind["swing_l"][i]
    o_i = ind["o"][i]

    bull_imb = (c_i > o_i) and (not np.isnan(sh)) and (c_i > sh)   # bullish → short fade
    bear_imb = (c_i < o_i) and (not np.isnan(sl)) and (c_i < sl)   # bearish → long fade

    if not bull_imb and not bear_imb:
        return None, None, None, None
    if bull_imb and bear_imb:
        return None, None, None, None   # ambiguous, skip

    direction = "short" if bull_imb else "long"

    # ── SL / TP geometry (identical to backtest run_sequential) ──────────────
    sl_atr = params["sl_atr_mult"]
    if direction == "long":
        sl_price = float(ind["l"][i]) - sl_atr * atr
    else:
        sl_price = float(ind["h"][i]) + sl_atr * atr

    tp_price = (float(o_i) + float(c_i)) / 2.0   # imbalance midpoint

    # Entry price ≈ next bar open. We use current bid/ask as proxy for gate checks.
    # Actual fill price is determined at order send time.
    # For gate checks use close of signal bar as conservative entry estimate.
    ep_est = c_i

    sl_dist = max(abs(ep_est - sl_price), atr * 0.05)
    tp_dist = abs(ep_est - tp_price)
    eff_rr  = tp_dist / sl_dist

    # ── min_rr gate ───────────────────────────────────────────────────────────
    if eff_rr < params["min_rr"]:
        return None, None, None, None

    # ── direction_ok gate ─────────────────────────────────────────────────────
    if direction == "long"  and tp_price < ep_est:
        return None, None, None, None
    if direction == "short" and tp_price > ep_est:
        return None, None, None, None

    return direction, sl_price, tp_price, eff_rr


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ORDER EXECUTION (from Septillion, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_lot_size(symbol, entry_price, sl_price, balance):
    """Identical to Septillion compute_lot_size."""
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
    """Identical to Septillion send_entry_order."""
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
        logger.error(f"[{symbol}] Entry FAILED retcode={getattr(result,'retcode',None)} "
                     f"{getattr(result,'comment','')}")
        return None
    logger.info(f"[{symbol}] ENTRY {direction.upper()} lot={lot} "
                f"price={price:.5f} sl={sl_price:.5f} tp={tp_price:.5f} "
                f"ticket={result.order}")
    return result


def send_close_order(symbol, position):
    """Identical to Septillion send_close_order."""
    otype = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY \
            else mt5.ORDER_TYPE_BUY
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
        "comment":      "oif_maxhold",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"[{symbol}] Close FAILED retcode={getattr(result,'retcode',None)}")
        return False
    logger.info(f"[{symbol}] FORCE CLOSED (max hold) ticket={position.ticket} "
                f"price={price:.5f}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — STATE RECONSTRUCTION (from Septillion, adapted for OIF)
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_state(symbol, position):
    """Recover trade state on engine restart from an open position."""
    entry_time = datetime.datetime.fromtimestamp(position.time,
                                                  tz=datetime.timezone.utc)
    now_utc    = datetime.datetime.now(tz=datetime.timezone.utc)
    bars_held  = int((now_utc - entry_time).total_seconds() / 300)  # M5 = 300s
    direction  = "long" if position.type == mt5.ORDER_TYPE_BUY else "short"
    sl_price   = position.sl
    ep         = position.price_open
    sl_dist    = abs(ep - sl_price) if sl_price else 0.0001

    logger.info(f"[{symbol}] RESTART: recovered {direction.upper()} "
                f"entry={ep:.5f} sl={sl_price:.5f} tp={position.tp:.5f} "
                f"bars_held≈{bars_held}")
    return {
        "direction":  direction,
        "entry_price":ep,
        "sl_price":   sl_price,
        "tp_price":   position.tp,
        "sl_dist":    sl_dist,
        "lot":        position.volume,
        "bars_held":  bars_held,
        "ticket":     position.ticket,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PER-SYMBOL STATE
# ══════════════════════════════════════════════════════════════════════════════

def make_symbol_state():
    return {
        "state":                 STATE_FLAT,
        "trade_state":           None,
        "bars_since_last_signal": COOLDOWN_BARS_MAX,  # start ready to trade
    }

COOLDOWN_BARS_MAX = max(p["cooldown_bars"] for p in SYMBOL_PARAMS.values())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — METRICS (from Septillion, simplified for single window)
# ══════════════════════════════════════════════════════════════════════════════

class Metrics:
    def __init__(self):
        self.data    = {sym: {"n": 0, "wins": 0, "total_r": 0.0}
                        for sym in PAIRS}
        self.peak    = None
        self.max_dd  = 0.0
        self._last_h = None

    def record(self, sym, r, balance):
        d = self.data[sym]
        d["n"]      += 1
        d["wins"]   += 1 if r > 0 else 0
        d["total_r"] += r
        if self.peak is None or balance > self.peak:
            self.peak = balance
        if self.peak and self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - balance) / self.peak)

    def _print(self, balance):
        tot_n = sum(d["n"] for d in self.data.values())
        tot_w = sum(d["wins"] for d in self.data.values())
        tot_r = sum(d["total_r"] for d in self.data.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"[OIF HOURLY REPORT]")
        logger.info(f"  Equity    : {balance:,.2f}")
        logger.info(f"  Trades    : {tot_n}  WR={tot_w/tot_n:.1%}  "
                    f"E={tot_r/tot_n:+.3f}R  totalR={tot_r:+.1f}"
                    if tot_n else "  Trades: 0")
        logger.info(f"  MaxDD     : {self.max_dd:.1%}")
        logger.info(f"  {'SYM':<8} {'n':>4} {'WR':>6} {'E':>7} {'totalR':>8}")
        for sym, d in self.data.items():
            if d["n"] > 0:
                logger.info(f"  {sym:<8} {d['n']:>4} "
                             f"{d['wins']/d['n']:>6.1%} "
                             f"{d['total_r']/d['n']:>+7.3f} "
                             f"{d['total_r']:>+8.2f}")
        logger.info('='*60)

    def check_hourly(self, balance):
        h = datetime.datetime.now(datetime.timezone.utc).hour
        if self._last_h is None:
            self._last_h = h
        if h != self._last_h:
            self._last_h = h
            self._print(balance)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — DATA FETCH + BAR CLOCK (from Septillion, adapted for M5)
# ══════════════════════════════════════════════════════════════════════════════

_CLOCK_SYM = PAIRS[2]   # EURUSD as bar clock reference

def fetch_bars(symbol):
    """Fetch last FETCH_BARS closed M5 bars. Drop the forming bar (index -1)."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, FETCH_BARS + 1)
    if rates is None or len(rates) < WARMUP + 10:
        logger.warning(f"[{symbol}] fetch failed or too few bars: "
                       f"{len(rates) if rates is not None else 'None'}")
        return None
    df = pd.DataFrame(rates)
    df["time_utc"] = pd.to_datetime(df["time"].astype(np.int64), unit="s",
                                     utc=True)
    df = df.iloc[:-1]   # drop forming bar — identical to backtest
    return df.reset_index(drop=True)


def get_last_closed_bar_time():
    rates = mt5.copy_rates_from_pos(_CLOCK_SYM, mt5.TIMEFRAME_M5, 0, 2)
    if rates is not None and len(rates) >= 2:
        return pd.Timestamp(rates[1]["time"], unit="s", tz="UTC")
    return None


def wait_for_new_bar(last_bar_time):
    """Block until M5 bar closes. Poll every 5s (from Septillion)."""
    while True:
        t = get_last_closed_bar_time()
        if t is not None and t > last_bar_time:
            return t
        time.sleep(5)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PROCESS ONE SYMBOL PER BAR
# ══════════════════════════════════════════════════════════════════════════════

def process_symbol(sym, sym_st, metrics, balance, bar_count):
    params = SYMBOL_PARAMS[sym]

    # ── Fetch fresh bars ──────────────────────────────────────────────────────
    df = fetch_bars(sym)
    if df is None:
        return

    ind = build_indicators(df)
    i   = ind["n"] - 1   # index of last closed bar

    # ── Broker cross-check (from Septillion) ──────────────────────────────────
    positions = mt5.positions_get(symbol=sym)
    positions = [p for p in positions
                 if p.magic == MAGIC and (p.comment or "").startswith(COMMENT)] \
                if positions else []

    # Detect server-side close (SL/TP hit)
    if not positions and sym_st["state"] == STATE_IN_POSITION:
        ts = sym_st["trade_state"]
        logger.info(f"[{sym}] Position closed server-side (SL/TP)")
        # Recover R from deal history
        r_mult = 0.0
        try:
            deals = mt5.history_deals_get(position=ts["ticket"])
            if deals and len(deals) >= 2:
                ep       = ts["entry_price"]
                sl_dist  = ts["sl_dist"]
                close_p  = deals[-1].price
                sign     = 1.0 if ts["direction"] == "long" else -1.0
                r_mult   = sign * (close_p - ep) / sl_dist if sl_dist > 0 else 0.0
        except Exception as ex:
            logger.warning(f"[{sym}] Could not recover R from deals: {ex}")
        metrics.record(sym, r_mult, balance)
        logger.info(f"[{sym}] R={r_mult:+.3f}")
        sym_st["state"]      = STATE_FLAT
        sym_st["trade_state"] = None
        sym_st["bars_since_last_signal"] = 0   # reset cooldown after trade

    # Detect orphaned position (restart recovery)
    if positions and sym_st["state"] == STATE_FLAT:
        logger.warning(f"[{sym}] Orphaned position found — recovering state")
        sym_st["trade_state"] = reconstruct_state(sym, positions[0])
        sym_st["state"]       = STATE_IN_POSITION

    # ── Max hold check ────────────────────────────────────────────────────────
    # In BT: trade expires at entry_bar + MAX_HOLD. Live: count bars held.
    if sym_st["state"] == STATE_IN_POSITION:
        ts = sym_st["trade_state"]
        ts["bars_held"] += 1
        if ts["bars_held"] >= MAX_HOLD:
            logger.info(f"[{sym}] MAX HOLD reached ({MAX_HOLD} bars) — force close")
            positions = mt5.positions_get(symbol=sym)
            positions = [p for p in positions
                         if p.magic == MAGIC and (p.comment or "").startswith(COMMENT)] \
                        if positions else []
            if positions:
                closed = send_close_order(sym, positions[0])
                if closed:
                    close_p = df["close"].iloc[-1]
                    ep      = ts["entry_price"]
                    sl_dist = ts["sl_dist"]
                    sign    = 1.0 if ts["direction"] == "long" else -1.0
                    r_mult  = sign * (close_p - ep) / sl_dist if sl_dist > 0 else 0.0
                    metrics.record(sym, r_mult, balance)
                    logger.info(f"[{sym}] R={r_mult:+.3f} (max hold)")
            sym_st["state"]       = STATE_FLAT
            sym_st["trade_state"] = None
            sym_st["bars_since_last_signal"] = 0

    # ── Entry logic ───────────────────────────────────────────────────────────
    if sym_st["state"] == STATE_FLAT:
        sym_st["bars_since_last_signal"] += 1

        direction, sl_price, tp_price, eff_rr = check_signal(
            ind, params, sym_st["bars_since_last_signal"]
        )

        if direction is not None:
            # Re-check broker has no position (race guard)
            positions = mt5.positions_get(symbol=sym)
            positions = [p for p in positions
                         if p.magic == MAGIC and (p.comment or "").startswith(COMMENT)] \
                        if positions else []
            if positions:
                logger.warning(f"[{sym}] Signal fired but position exists — skipping")
            else:
                tick = mt5.symbol_info_tick(sym)
                ep   = tick.ask if direction == "long" else tick.bid

                # Re-evaluate direction_ok and min_rr against ACTUAL entry price
                atr_e   = ind["atr14"][i]
                sl_dist = max(abs(ep - sl_price), atr_e * 0.05)
                tp_dist = abs(ep - tp_price)
                eff_rr_actual = tp_dist / sl_dist

                dir_ok = (tp_price >= ep) if direction == "long" else (tp_price <= ep)

                if not dir_ok:
                    logger.info(f"[{sym}] Signal SKIPPED: TP wrong side at actual fill "
                                f"(tp={tp_price:.5f} ep={ep:.5f} dir={direction})")
                elif eff_rr_actual < params["min_rr"]:
                    logger.info(f"[{sym}] Signal SKIPPED: eff_rr={eff_rr_actual:.3f} "
                                f"< min_rr={params['min_rr']}")
                else:
                    lot = compute_lot_size(sym, ep, sl_price, balance)
                    if lot is None:
                        logger.error(f"[{sym}] Lot calc failed")
                    else:
                        result = send_entry_order(sym, direction, sl_price, tp_price, lot)
                        if result:
                            time.sleep(0.5)
                            # Confirm position opened
                            pos_new = mt5.positions_get(symbol=sym)
                            pos_new = [p for p in pos_new
                                       if p.magic == MAGIC
                                       and (p.comment or "").startswith(COMMENT)] \
                                      if pos_new else []
                            if pos_new:
                                ap     = pos_new[0].price_open
                                asl    = pos_new[0].sl
                                atp    = pos_new[0].tp
                                adist  = abs(ap - asl)
                            else:
                                ap    = ep
                                asl   = sl_price
                                atp   = tp_price
                                adist = sl_dist

                            sym_st["trade_state"] = {
                                "direction":   direction,
                                "entry_price": ap,
                                "sl_price":    asl,
                                "tp_price":    atp,
                                "sl_dist":     adist,
                                "lot":         lot,
                                "bars_held":   0,
                                "ticket":      result.order,
                            }
                            sym_st["state"] = STATE_IN_POSITION
                            sym_st["bars_since_last_signal"] = 0

                            logger.info(
                                f"[{sym}] TRADE OPEN: {direction.upper()} "
                                f"entry={ap:.5f} sl={asl:.5f} tp={atp:.5f} "
                                f"eff_rr={eff_rr_actual:.2f} lot={lot}"
                            )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — SYMBOL DIAGNOSTIC (from Septillion)
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostic():
    logger.info("=== SYMBOL DIAGNOSTIC ===")
    for sym in PAIRS:
        info = mt5.symbol_info(sym)
        if info is None:
            avail = mt5.symbols_get() or []
            cands = [s.name for s in avail
                     if sym[:3] in s.name or sym[3:] in s.name][:8]
            logger.warning(f"  {sym}: NOT FOUND. Candidates: {cands}")
        else:
            if not info.visible:
                mt5.symbol_select(sym, True)
            tick  = mt5.symbol_info_tick(sym)
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 5)
            logger.info(
                f"  {sym}: visible={info.visible}  spread={info.spread}  "
                f"digits={info.digits}  "
                f"tick={'OK' if tick else 'None'}  "
                f"bars={'OK len='+str(len(rates)) if rates is not None else 'None'}"
            )
    logger.info("=== END DIAGNOSTIC ===")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_live():
    if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN,
                          password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    acct = mt5.account_info()
    logger.info(f"MT5 connected | account={acct.login} | balance={acct.balance:.2f}")
    logger.info(f"Engine: OIF LIVE | Magic: {MAGIC}")
    logger.info(f"Symbols: {PAIRS}")
    logger.info(f"Risk: {RISK_PER_TRADE:.1%} per trade | MAX_HOLD: {MAX_HOLD} bars")
    logger.info("Params:")
    for sym, p in SYMBOL_PARAMS.items():
        logger.info(f"  {sym}: sl={p['sl_atr_mult']} body={p['body_atr_mult']} "
                    f"ext={p['ext_atr_mult']} mrr={p['min_rr']} "
                    f"rvp={p['rv_pct_thresh']} cd={p['cooldown_bars']}")
    logger.info("=" * 60)

    run_diagnostic()

    # ── Per-symbol state ──────────────────────────────────────────────────────
    sym_states = {sym: make_symbol_state() for sym in PAIRS}
    metrics    = Metrics()
    bar_count  = 0

    # ── Startup: recover open positions ──────────────────────────────────────
    for sym, st in sym_states.items():
        positions = mt5.positions_get(symbol=sym)
        if positions:
            for pos in positions:
                if pos.magic == MAGIC and (pos.comment or "").startswith(COMMENT):
                    st["trade_state"] = reconstruct_state(sym, pos)
                    st["state"]       = STATE_IN_POSITION
                    break

    # ── Seed bar clock ────────────────────────────────────────────────────────
    last_bar_time = get_last_closed_bar_time()
    if last_bar_time is None:
        last_bar_time = pd.Timestamp.utcnow()
    logger.info(f"Bar clock seeded: {last_bar_time} UTC — waiting for first M5 close...")

    while True:
        try:
            new_bar_time  = wait_for_new_bar(last_bar_time)
            last_bar_time = new_bar_time
            bar_count    += 1

            logger.info(f"── BAR {bar_count} | {new_bar_time} UTC "
                        f"──────────────────────────────")

            balance = mt5.account_info().balance

            for sym in PAIRS:
                process_symbol(sym, sym_states[sym], metrics, balance, bar_count)

            metrics.check_hourly(balance)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
            break
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(30)

    mt5.shutdown()
    logger.info("OIF Live engine stopped.")
    metrics._print(mt5.account_info().balance if mt5.account_info() else 0)


if __name__ == "__main__":
    run_live()
