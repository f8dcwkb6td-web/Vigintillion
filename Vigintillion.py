"""
==============================================================================
M1 Microstructure Scalp v2  |  USDJPY  |  LIVE TRADING ENGINE
==============================================================================
Companion to micro_scalp_v2.py (backtest).

Behaviour is IDENTICAL to backtest:
  - Signal logic:    same indicator functions, same BEST_PARAMS
  - SL/TP:          same sweep-based placement
  - Early exit:     VWAP adverse cross + 3 consecutive adverse candles
  - Max hold:       60 bars
  - Session gates:  London 07-10 UTC, NY 13-15 UTC
  - Force-close:    at session boundary
  - Cooldown:       10 bars between entries
  - Max trades:     3 per session

ARCHITECTURE:
  Single-threaded, M1-bar-close driven event loop.
  No ticks.  No async.  No ML.  No multiprocessing.

STARTUP RECOVERY:
  Detects open MT5 position on startup and reconstructs state.
  Safe to restart on VPS without orphaning trades.
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
    print("ERROR: MetaTrader5 package not installed. pip install MetaTrader5")
    sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("LIVE_V2")
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler("live_scalp.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
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

# ── Strategy constants (must match backtest) ──────────────────────────────────
SYMBOL                 = "USDJPY"
RISK_PER_TRADE         = 0.06       # 1% risk per trade
MAGIC                  = 20260226
COMMENT                = "micro_v2_live"
MAX_HOLD               = 60         # bars
VWAP_WINDOW            = 10
SL_MULTIPLIER          = 1.0
COOLDOWN_BARS          = 10
MAX_TRADES_PER_SESSION = 3
WARMUP_BARS            = 500000       # minimum bars needed for indicators

# ── Best params from optimisation 2026-02-26 ─────────────────────────────────
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

# ── State machine ─────────────────────────────────────────────────────────────
STATE_FLAT        = "FLAT"
STATE_IN_POSITION = "IN_POSITION"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — INDICATORS  (identical to backtest)
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
#  SECTION 2 — DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_bars(n=500000):
    """Fetch last n closed M1 bars. Returns DataFrame or None."""
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, n + 1)
    if rates is None:
        logger.warning(f"fetch_bars: mt5 returned None — error={mt5.last_error()}")
        return None
    if len(rates) < WARMUP_BARS:
        logger.warning(f"fetch_bars: only {len(rates)} bars returned, need {WARMUP_BARS} — still warming up")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.iloc[:-1]   # drop current (forming) bar — only closed bars


def get_account_balance():
    info = mt5.account_info()
    return info.balance if info else None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — INDICATOR COMPUTATION (on recent bars)
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df):
    """
    Compute all indicators needed for signal + early-exit on the last N bars.
    Returns dict of arrays aligned to df index.
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["tick_volume"].values

    ind = {}
    ind["o"] = o; ind["h"] = h; ind["l"] = l; ind["c"] = c; ind["v"] = v

    ind["atr14"]   = compute_atr(h, l, c, 14)
    ind["rvol_30"] = rolling_realized_vol(c, 30)

    VOL_LB = min(28_800, len(c) // 2)
    for q in [40, 50, 60]:
        ind[f"rvol_q{q}"] = rolling_quantile(ind["rvol_30"], VOL_LB, q / 100)

    bar_range = h - l
    ind["bar_range"] = bar_range
    SPR_LB = min(43_200, len(c) // 2)
    for q in [20, 30, 40]:
        ind[f"spread_q{q}"] = rolling_quantile(bar_range, SPR_LB, q / 100)

    ind["vol_mean_60"] = pd.Series(v.astype(float)).rolling(60, min_periods=10).mean().values
    ind["vwap"]        = micro_vwap(h, l, c, v, VWAP_WINDOW)

    body = np.abs(c - o); rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        ind["body_ratio"] = np.where(rng > 0, body / rng, 0.0)
        ind["close_pos"]  = np.where(rng > 0, (c - l) / rng, 0.5)

    # Session flags
    dt    = pd.DatetimeIndex(df["time"])
    hours = dt.hour
    ind["in_london"] = (hours >= 7)  & (hours < 10)
    ind["in_ny"]     = (hours >= 13) & (hours < 15)
    ind["in_window"] = ind["in_london"] | ind["in_ny"]
    ind["times"]     = df["time"].values
    ind["dates"]     = np.array(dt.date)

    return ind


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — ASIAN RANGE
# ══════════════════════════════════════════════════════════════════════════════

class AsianRangeTracker:
    """Maintains today's and yesterday's Asian session range."""

    def __init__(self):
        self.today_date   = None
        self.today_hi     = -np.inf
        self.today_lo     = +np.inf
        self.prev_hi      = np.nan
        self.prev_lo      = np.nan

    def update(self, bar_time, bar_h, bar_l):
        d    = bar_time.date() if hasattr(bar_time, "date") else datetime.datetime.utcfromtimestamp(bar_time.astype("int64") // 1_000_000_000).date()
        hour = bar_time.hour   if hasattr(bar_time, "hour")  else datetime.datetime.utcfromtimestamp(bar_time.astype("int64") // 1_000_000_000).hour

        if d != self.today_date:
            # Day rolled — yesterday's range is now previous
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
        """Return (prev_hi, prev_lo) — valid Asian range for trade window."""
        return self.prev_hi, self.prev_lo


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SIGNAL DETECTION (last bar only)
# ══════════════════════════════════════════════════════════════════════════════

def check_entry_signal(ind, asian_hi, asian_lo, params, last_exit_bar, sess_count_today):
    """
    Evaluate signal on the most recent closed bar (index -1).
    Returns ("long" | "short" | None, sl_price, tp_price, sl_dist)
    """
    i    = len(ind["c"]) - 1
    p    = params

    # ── Session gate ─────────────────────────────────────────────────────
    if not ind["in_window"][i]:
        return None, None, None, None

    # ── Cooldown gate ─────────────────────────────────────────────────────
    if i - last_exit_bar < COOLDOWN_BARS:
        return None, None, None, None

    # ── Session trade count gate ──────────────────────────────────────────
    if sess_count_today >= MAX_TRADES_PER_SESSION:
        return None, None, None, None

    # ── Regime ───────────────────────────────────────────────────────────
    vq = p["vol_threshold_q"].replace("q", "")
    sq = p["spread_q"].replace("q", "")
    regime_ok = (ind["rvol_30"][i] <= ind[f"rvol_q{vq}"][i]) and \
                (ind["bar_range"][i] <= ind[f"spread_q{sq}"][i])
    if not regime_ok:
        return None, None, None, None

    # ── Sweep detection ───────────────────────────────────────────────────
    N    = p["sweep_lookback"]
    mult = p["sweep_atr_mult"]
    atr  = ind["atr14"][i]

    start = max(0, i - N)
    roll_lo = ind["l"][start:i].min() if i > start else np.nan
    roll_hi = ind["h"][start:i].max() if i > start else np.nan

    h_i = ind["h"][i]; l_i = ind["l"][i]; c_i = ind["c"][i]

    bull_gen   = (not np.isnan(roll_lo)) and (l_i < roll_lo - mult * atr) and (c_i > roll_lo)
    bear_gen   = (not np.isnan(roll_hi)) and (h_i > roll_hi + mult * atr) and (c_i < roll_hi)

    has_range  = not (np.isnan(asian_hi) or np.isnan(asian_lo))
    bull_asian = has_range and (l_i < asian_lo - mult * atr * 0.5) and (c_i > asian_lo)
    bear_asian = has_range and (h_i > asian_hi + mult * atr * 0.5) and (c_i < asian_hi)

    bull_sweep = bull_gen or bull_asian
    bear_sweep = bear_gen or bear_asian

    if not (bull_sweep or bear_sweep):
        return None, None, None, None

    # ── Displacement ──────────────────────────────────────────────────────
    o_i    = ind["o"][i]
    body   = abs(c_i - o_i)
    disp_A_bull = (c_i > o_i) and (body >= p["body_atr_mult"] * atr) and (ind["body_ratio"][i] >= 0.70)
    disp_A_bear = (c_i < o_i) and (body >= p["body_atr_mult"] * atr) and (ind["body_ratio"][i] >= 0.70)
    vol_spike   = ind["v"][i] >= p["vol_mult"] * ind["vol_mean_60"][i]
    disp_B_bull = vol_spike and (ind["close_pos"][i] >= 0.80)
    disp_B_bear = vol_spike and (ind["close_pos"][i] <= 0.20)

    disp_bull = disp_A_bull or disp_B_bull
    disp_bear = disp_A_bear or disp_B_bear

    # Check current or previous bar for displacement
    if i > 0:
        o_p = ind["o"][i-1]; c_p = ind["c"][i-1]; body_p = abs(c_p - o_p)
        h_p = ind["h"][i-1]; l_p = ind["l"][i-1]
        br_p = (body_p / (h_p - l_p)) if (h_p - l_p) > 0 else 0
        cp_p = ((c_p - l_p) / (h_p - l_p)) if (h_p - l_p) > 0 else 0.5
        atr_p = ind["atr14"][i-1]
        vol_spike_p = ind["v"][i-1] >= p["vol_mult"] * ind["vol_mean_60"][i-1]
        disp_A_bull_p = (c_p > o_p) and (body_p >= p["body_atr_mult"] * atr_p) and (br_p >= 0.70)
        disp_A_bear_p = (c_p < o_p) and (body_p >= p["body_atr_mult"] * atr_p) and (br_p >= 0.70)
        disp_B_bull_p = vol_spike_p and (cp_p >= 0.80)
        disp_B_bear_p = vol_spike_p and (cp_p <= 0.20)
        disp_bull = disp_bull or disp_A_bull_p or disp_B_bull_p
        disp_bear = disp_bear or disp_A_bear_p or disp_B_bear_p

    # ── Final signal ──────────────────────────────────────────────────────
    long_cond  = bull_sweep and disp_bull
    short_cond = bear_sweep and disp_bear

    if not (long_cond or short_cond):
        return None, None, None, None

    # Resolve conflict: both — skip
    if long_cond and short_cond:
        return None, None, None, None

    direction = "long" if long_cond else "short"
    buf       = p["buffer_atr"]
    rr        = p["rr_ratio"]

    # SL placement (next bar entry price not known yet — use current close as proxy)
    # Actual entry is next bar open; SL/TP recomputed after fill
    if direction == "long":
        sl_ref   = roll_lo if (not np.isnan(roll_lo)) else None
        sl_price = (sl_ref - buf * atr) if sl_ref else (c_i - SL_MULTIPLIER * atr)
    else:
        sl_ref   = roll_hi if (not np.isnan(roll_hi)) else None
        sl_price = (sl_ref + buf * atr) if sl_ref else (c_i + SL_MULTIPLIER * atr)

    return direction, sl_price, rr, atr


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — EARLY EXIT EVALUATION (bar-close)
# ══════════════════════════════════════════════════════════════════════════════

def check_early_exit(ind, trade_state):
    """
    Evaluate early exit conditions on the last closed bar.
    Returns True if should close now.
    """
    i   = len(ind["c"]) - 1
    dir = trade_state["direction"]

    # ── Max hold ──────────────────────────────────────────────────────────
    trade_state["hold_count"] += 1
    if trade_state["hold_count"] >= MAX_HOLD:
        logger.info(f"  EXIT: max hold reached ({MAX_HOLD} bars)")
        return True

    # ── Session force-close ───────────────────────────────────────────────
    if not ind["in_window"][i]:
        logger.info("  EXIT: session boundary")
        return True

    hc = trade_state["hold_count"]

    # ── E1: VWAP adverse cross ────────────────────────────────────────────
    if i >= 1:
        cur_c    = ind["c"][i];   cur_vwap  = ind["vwap"][i]
        prev_c   = ind["c"][i-1]; prev_vwap = ind["vwap"][i-1]
        if dir == "long"  and (cur_c < cur_vwap)  and (prev_c >= prev_vwap) and hc >= 3:
            logger.info("  EXIT: VWAP adverse cross (long)")
            return True
        if dir == "short" and (cur_c > cur_vwap)  and (prev_c <= prev_vwap) and hc >= 3:
            logger.info("  EXIT: VWAP adverse cross (short)")
            return True

    # ── E2: 3 consecutive adverse candles ────────────────────────────────
    c_i = ind["c"][i]; o_i = ind["o"][i]
    adverse = (c_i < o_i) if dir == "long" else (c_i > o_i)

    if adverse:
        trade_state["consec_adverse"] += 1
    else:
        trade_state["consec_adverse"] = 0

    if trade_state["consec_adverse"] >= 3 and hc >= 3:
        logger.info(f"  EXIT: 3 consecutive adverse candles")
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_lot_size(entry_price, sl_price, balance):
    """Risk-based lot size using broker contract specs."""
    sym = mt5.symbol_info(SYMBOL)
    if sym is None:
        logger.error("symbol_info returned None")
        return None

    tick_value        = sym.trade_tick_value
    tick_size         = sym.trade_tick_size
    pip_value_per_lot = tick_value / tick_size

    risk_amount    = balance * RISK_PER_TRADE
    stop_distance  = abs(entry_price - sl_price)

    if stop_distance < 1e-9:
        logger.error("Stop distance is zero — lot calc aborted")
        return None

    raw_lot = risk_amount / (stop_distance * pip_value_per_lot)
    lot     = max(sym.volume_min,
                  min(sym.volume_max,
                      round(raw_lot / sym.volume_step) * sym.volume_step))
    return lot


def send_entry_order(direction, sl_price, tp_price, lot):
    """Send market order with server-side SL/TP. Returns result or None."""
    tick  = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if direction == "long" else tick.bid
    otype = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      SYMBOL,
        "volume":      lot,
        "type":        otype,
        "price":       price,
        "sl":          sl_price,
        "tp":          tp_price,
        "deviation":   10,
        "magic":       MAGIC,
        "comment":     COMMENT,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Entry order FAILED: retcode={getattr(result,'retcode',None)} {getattr(result,'comment','')}")
        return None
    logger.info(f"ENTRY {direction.upper()} | lot={lot} price={price:.3f} sl={sl_price:.3f} tp={tp_price:.3f} ticket={result.order}")
    return result


def send_close_order(position):
    """Close the given position with a market order."""
    otype = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick  = mt5.symbol_info_tick(SYMBOL)
    price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      SYMBOL,
        "volume":      position.volume,
        "type":        otype,
        "position":    position.ticket,
        "price":       price,
        "deviation":   10,
        "magic":       MAGIC,
        "comment":     "early_exit",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Close order FAILED: retcode={getattr(result,'retcode',None)} {getattr(result,'comment','')}")
        return False
    logger.info(f"CLOSED ticket={position.ticket} price={price:.3f}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — TRADE STATE RECONSTRUCTION (VPS restart recovery)
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_state(position, df):
    """
    Rebuild trade_state from an open MT5 position.
    Uses entry time to compute hold_count.
    """
    entry_time = datetime.datetime.fromtimestamp(position.time, tz=datetime.timezone.utc)
    now_utc    = datetime.datetime.now(tz=datetime.timezone.utc)

    # Estimate hold count from elapsed minutes (each M1 bar = 1 minute)
    elapsed_minutes = int((now_utc - entry_time).total_seconds() / 60)
    hold_count      = max(0, elapsed_minutes)

    direction = "long" if position.type == mt5.ORDER_TYPE_BUY else "short"
    sl_price  = position.sl
    ep        = position.price_open
    sl_dist   = abs(ep - sl_price) if sl_price else 0.01

    # Estimate consecutive adverse candles from recent bars
    consec = 0
    o_arr  = df["open"].values
    c_arr  = df["close"].values
    for k in range(min(3, len(df))):
        idx = -(k + 1)
        adv = (c_arr[idx] < o_arr[idx]) if direction == "long" else (c_arr[idx] > o_arr[idx])
        if adv:
            consec += 1
        else:
            break

    state = {
        "direction":     direction,
        "entry_price":   ep,
        "sl_price":      sl_price,
        "tp_price":      position.tp,
        "sl_dist":       sl_dist,
        "lot":           position.volume,
        "risk_amount":   sl_dist * (position.volume * mt5.symbol_info(SYMBOL).trade_tick_value / mt5.symbol_info(SYMBOL).trade_tick_size),
        "hold_count":    hold_count,
        "consec_adverse": consec,
        "ticket":        position.ticket,
        "bar_index_entry": len(df) - 1 - hold_count,
    }
    logger.info(f"RECOVERED position: dir={direction} entry={ep:.3f} hold={hold_count}bars consec_adv={consec}")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — METRICS
# ══════════════════════════════════════════════════════════════════════════════

class Metrics:
    def __init__(self):
        self.total_trades  = 0
        self.wins          = 0
        self.losses        = 0
        self.total_r       = 0.0
        self.equity_peak   = None
        self.max_dd        = 0.0
        self.last_hour     = None

    def record_trade(self, r_multiple, balance):
        self.total_trades += 1
        if r_multiple > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.total_r += r_multiple

        if self.equity_peak is None or balance > self.equity_peak:
            self.equity_peak = balance
        dd = (self.equity_peak - balance) / self.equity_peak if self.equity_peak > 0 else 0.0
        self.max_dd = max(self.max_dd, dd)

    def hourly_report(self, balance):
        wr  = self.wins / self.total_trades if self.total_trades else 0.0
        exp = self.total_r / self.total_trades if self.total_trades else 0.0
        logger.info(
            f"\n[HOURLY REPORT]\n"
            f"  Trades:      {self.total_trades}\n"
            f"  WR:          {wr:.1%}\n"
            f"  Expectancy:  {exp:+.2f}R\n"
            f"  Total R:     {self.total_r:+.1f}\n"
            f"  Max DD:      {self.max_dd:.1%}\n"
            f"  Equity:      {balance:,.2f}"
        )

    def check_hourly(self, balance):
        now_h = datetime.datetime.now(datetime.timezone.utc).hour
        if self.last_hour is None:
            self.last_hour = now_h
        if now_h != self.last_hour:
            self.last_hour = now_h
            self.hourly_report(balance)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — MAIN EVENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def wait_for_new_bar(last_bar_time):
    """
    Block until a new M1 bar has closed.
    Polls every 5 seconds. Returns when bar_time > last_bar_time.
    """
    while True:
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 1, 1)
        if rates is not None and len(rates) > 0:
            t = pd.Timestamp(rates[0]["time"], unit="s")
            if t > last_bar_time:
                return t
        time.sleep(5)


def run_live():
    # ── Connect ───────────────────────────────────────────────────────────
    if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN, password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    logger.info(f"MT5 connected | account={mt5.account_info().login} | balance={mt5.account_info().balance:.2f}")

    # ── Init state ────────────────────────────────────────────────────────
    state          = STATE_FLAT
    trade_state    = None
    last_exit_bar  = -COOLDOWN_BARS - 1
    sess_date      = None
    sess_count     = 0
    asian_tracker  = AsianRangeTracker()
    metrics        = Metrics()

    # ── Startup: detect existing position and reconstruct ─────────────────
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions:
        pos = positions[0]
        if pos.magic == MAGIC:
            df_init = fetch_bars(500)
            if df_init is not None:
                trade_state = reconstruct_state(pos, df_init)
                state       = STATE_IN_POSITION
                logger.info("STARTUP: recovered existing trade — entering IN_POSITION")
        else:
            logger.warning(f"STARTUP: found position with wrong magic {pos.magic} — treating as FLAT")

    # ── Seed last bar time ────────────────────────────────────────────────
    rates_init = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 1, 1)
    last_bar_time = pd.Timestamp(rates_init[0]["time"], unit="s") if rates_init is not None else pd.Timestamp.now(tz="UTC")

    logger.info(f"Live engine started | state={state}")
    logger.info("=" * 70)

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        try:
            # 1. Wait for new closed M1 bar
            new_bar_time  = wait_for_new_bar(last_bar_time)
            last_bar_time = new_bar_time

            # 2. Fetch recent bars and compute indicators
            df  = fetch_bars(700000)
            if df is None:
                logger.warning("fetch_bars returned None — skipping bar")
                continue

            ind = compute_indicators(df)

            # 3. Update Asian range with this bar
            bar_ts = df["time"].iloc[-1]
            asian_tracker.update(bar_ts, df["high"].iloc[-1], df["low"].iloc[-1])
            asian_hi, asian_lo = asian_tracker.get()

            # 4. Reset session trade count on new day
            bar_date = df["time"].iloc[-1].date()
            if bar_date != sess_date:
                sess_date  = bar_date
                sess_count = 0

            # 5. Cross-check broker state (non-negotiable)
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions and state == STATE_FLAT:
                logger.warning("Broker has position but state=FLAT — correcting to IN_POSITION")
                pos = positions[0]
                if pos.magic == MAGIC:
                    trade_state = reconstruct_state(pos, df)
                    state       = STATE_IN_POSITION
            elif not positions and state == STATE_IN_POSITION:
                # SL or TP was hit server-side — record trade closure
                logger.info("Position closed server-side (SL/TP hit)")
                if trade_state:
                    balance   = mt5.account_info().balance
                    ep        = trade_state["entry_price"]
                    sl_dist   = trade_state["sl_dist"]
                    direction = trade_state["direction"]
                    # Get last closed deal to compute actual R
                    deals = mt5.history_deals_get(position=trade_state.get("ticket", 0))
                    if deals and len(deals) >= 2:
                        close_price = deals[-1].price
                        sign        = 1 if direction == "long" else -1
                        r_mult      = sign * (close_price - ep) / sl_dist if sl_dist > 0 else 0.0
                    else:
                        r_mult = 0.0
                    metrics.record_trade(r_mult, balance)
                    logger.info(f"TRADE CLOSED (server): R={r_mult:+.3f}")
                state          = STATE_FLAT
                trade_state    = None
                last_exit_bar  = len(df) - 1

            # 6. State machine
            balance = mt5.account_info().balance

            if state == STATE_FLAT:
                direction, sl_price, rr, atr = check_entry_signal(
                    ind, asian_hi, asian_lo, BEST_PARAMS, last_exit_bar, sess_count
                )
                if direction is not None:
                    # Re-check broker — hard stop against duplicate entry
                    positions = mt5.positions_get(symbol=SYMBOL)
                    if positions:
                        logger.warning("Skipping entry — position already exists at broker")
                    else:
                        # Lot size based on next-bar open proxy (current close)
                        c_last   = df["close"].iloc[-1]
                        ep_proxy = c_last  # actual fill = next bar open; SL computed from signal bar
                        lot      = compute_lot_size(ep_proxy, sl_price, balance)
                        if lot is None:
                            logger.error("Lot calc failed — skipping entry")
                        else:
                            # TP from SL distance
                            sl_dist  = abs(ep_proxy - sl_price)
                            tp_price = (ep_proxy + sl_dist * rr) if direction == "long" else (ep_proxy - sl_dist * rr)

                            result = send_entry_order(direction, sl_price, tp_price, lot)
                            if result:
                                # Get actual fill price
                                time.sleep(0.5)
                                pos_new = mt5.positions_get(symbol=SYMBOL)
                                if pos_new:
                                    actual_ep  = pos_new[0].price_open
                                    actual_sl  = pos_new[0].sl
                                    actual_tp  = pos_new[0].tp
                                    actual_dist = abs(actual_ep - actual_sl)
                                else:
                                    actual_ep   = ep_proxy
                                    actual_sl   = sl_price
                                    actual_tp   = tp_price
                                    actual_dist = sl_dist

                                trade_state = {
                                    "direction":      direction,
                                    "entry_price":    actual_ep,
                                    "sl_price":       actual_sl,
                                    "tp_price":       actual_tp,
                                    "sl_dist":        actual_dist,
                                    "lot":            lot,
                                    "risk_amount":    balance * RISK_PER_TRADE,
                                    "hold_count":     0,
                                    "consec_adverse": 0,
                                    "ticket":         result.order,
                                    "bar_index_entry": len(df) - 1,
                                }
                                state       = STATE_IN_POSITION
                                sess_count += 1

            elif state == STATE_IN_POSITION:
                should_exit = check_early_exit(ind, trade_state)
                if should_exit:
                    positions = mt5.positions_get(symbol=SYMBOL)
                    if positions:
                        closed = send_close_order(positions[0])
                        if closed:
                            # Compute R
                            close_price = df["close"].iloc[-1]
                            ep          = trade_state["entry_price"]
                            sl_dist     = trade_state["sl_dist"]
                            sign        = 1 if trade_state["direction"] == "long" else -1
                            r_mult      = sign * (close_price - ep) / sl_dist if sl_dist > 0 else 0.0
                            metrics.record_trade(r_mult, mt5.account_info().balance)
                            logger.info(f"TRADE CLOSED (early): R={r_mult:+.3f}")
                            state         = STATE_FLAT
                            trade_state   = None
                            last_exit_bar = len(df) - 1
                    else:
                        # Already closed server-side
                        state       = STATE_FLAT
                        trade_state = None

            # 7. Hourly metrics
            metrics.check_hourly(balance)

        except KeyboardInterrupt:
            logger.info("Shutdown requested — exiting cleanly")
            break
        except Exception as e:
            logger.exception(f"Unhandled error in main loop: {e}")
            time.sleep(30)   # brief pause before retry

    mt5.shutdown()
    logger.info("MT5 disconnected. Live engine stopped.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_live()
