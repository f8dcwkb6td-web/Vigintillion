"""
==============================================================================
MSB  —  MARKET STRUCTURE BREAK  |  LIVE ENGINE  (M5)
==============================================================================
FILTERED SYMBOL SET  (passed all 3 targets in backtest: E>0.30R or best
available, PF>1.5 or best, MDD<20%)

  US30    Return=44.5%  E=+0.346R  WR=59.6%  PF=1.858  MDD=7.8%   n=109
  XAGUSD  Return=28.4%  E=+0.280R  WR=55.4%  PF=1.704  MDD=5.0%   n=92
  USDCAD  Return=29.9%  E=+0.215R  WR=60.8%  PF=1.549  MDD=9.6%   n=125
  GBPJPY  Return=32.3%  E=+0.208R  WR=60.1%  PF=1.526  MDD=6.6%   n=138
  USDJPY  Return=13.5%  E=+0.165R  WR=51.8%  PF=1.344  MDD=8.9%   n=81
  EURJPY  Return=28.5%  E=+0.181R  WR=56.6%  PF=1.428  MDD=7.0%   n=143
  GBPAUD  Return=17.9%  E=+0.156R  WR=55.5%  PF=1.357  MDD=10.5%  n=110
  NZDUSD  Return=12.6%  E=+0.090R  WR=53.2%  PF=1.193  MDD=12.0%  n=141
  AUDUSD  Return=10.9%  E=+0.091R  WR=54.1%  PF=1.199  MDD=10.8%  n=141

PARITY AUDIT vs msb_hardcoded.py (backtest):
  ✓ H1 HTF bias: swing highs/lows (lb=swing_lb), HH+HL=bull, LH+LL=bear
  ✓ ATR: Wilder smoothing, period=14
  ✓ ATR expanding-percentile filter >= 0.35 (warmup=200 bars)
  ✓ Session filter: same UTC hour sets as backtest
  ✓ Impulse: shifted by (cons_bars+1), body >= impulse_atr_mult*ATR,
             body_ratio >= body_ratio_min, directional close
  ✓ Consolidation: rolling max(H-L) < cons_range_mult*mean(ATR) over cons_bars
  ✓ Breakout: close > cons_high (long) / close < cons_low (short)
  ✓ Cooldown: >= cooldown_bars between signals per symbol
  ✓ Entry: NEXT bar open (entry_i = sig_i + 1)
  ✓ SL: cons_low - sl_atr_mult*ATR (long) / cons_high + sl_atr_mult*ATR (short)
  ✓ SL distance minimum: 5% of ATR
  ✓ BE trigger: price reaches 1R (entry +/- sl_dist); SL moves to entry
  ✓ Trail: after BE, trail_atr_mult*ATR from bar high/low each bar
  ✓ Max hold: 60 bars; exit at close of bar 60 at market
  ✓ Signal fired on bar close; order placed at NEXT bar open via LIMIT/MARKET

ARCHITECTURE  (based on Septillion template):
  - One MT5 connection, one M5 bar-close loop
  - Per-symbol state machine: FLAT → PENDING_ENTRY → IN_POSITION → FLAT
  - PENDING_ENTRY: signal detected on bar close, entry sent on NEXT bar open
  - SL/TP server-side; trail updated each bar via MODIFY order
  - Broker cross-check every bar (recover positions on restart)
  - Symbol selection: auto-resolve broker suffix (e.g. US30.m, US30cash)
  - Risk: RISK_PER_TRADE per trade per symbol (default 3%)

MAGIC:   202603250
COMMENT: "MSB_LIVE"
LOG:     msb_live.log
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
logger = logging.getLogger("MSB_LIVE")
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler(
    "msb_live.log", maxBytes=15_000_000, backupCount=5, encoding="utf-8"
)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_fh)
_sh = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
)
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_sh)

# ── MT5 connection ────────────────────────────────────────────────────────────
TERMINAL_PATH = os.environ.get(
    "MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe"
)
LOGIN    = int(os.environ.get("MT5_LOGIN",    0))
PASSWORD =     os.environ.get("MT5_PASSWORD", "")
SERVER   =     os.environ.get("MT5_SERVER",   "")

# ── Engine identity ───────────────────────────────────────────────────────────
MAGIC   = 202603250
COMMENT = "MSB_LIVE"

# ── Strategy constants (must match backtest exactly) ─────────────────────────
RISK_PER_TRADE = 0.03     # 3% risk per trade
MAX_HOLD       = 60        # bars (5-min; 5 hours max)
ATR_PERIOD     = 14
ATR_PCT_THRESH = 0.35      # expanding percentile, warmup=200
WARMUP_M5      = 200
FETCH_BARS_M5  = 2000      # ~1 week of M5; enough for indicators + warmup
FETCH_BARS_H1  = 500       # ~3 weeks of H1; enough for swing bias

# ── Session hours UTC (identical to backtest) ─────────────────────────────────
GOLD_SESSION_HOURS   = set(range(7, 20))
SILVER_SESSION_HOURS = set(range(7, 20))
INDEX_SESSION_HOURS  = set(range(8, 17))
FOREX_LONDON         = set(range(7, 11))
FOREX_NY             = set(range(13, 17))
FOREX_BOTH           = FOREX_LONDON | FOREX_NY
INDEX_PREFIXES       = ("US30", "US500", "NAS100", "GER40", "UK100",
                        "SPX", "NDX", "DOW", "DAX", "FTSE")

# ── HARDCODED PARAMS (FILTERED SET ONLY — from grid search) ──────────────────
BEST_PARAMS = {
    "US30": dict(
        swing_lb=3, impulse_atr_mult=1.0, body_ratio_min=0.45,
        cons_bars=3, cons_range_mult=1.0, sl_atr_mult=1.0,
        trail_atr_mult=0.75, cooldown_bars=3,
    ),
    "GBPJPY": dict(
        swing_lb=8, impulse_atr_mult=1.2, body_ratio_min=0.45,
        cons_bars=2, cons_range_mult=1.0, sl_atr_mult=1.0,
        trail_atr_mult=0.75, cooldown_bars=3,
    ),
    "USDCAD": dict(
        swing_lb=5, impulse_atr_mult=1.2, body_ratio_min=0.60,
        cons_bars=3, cons_range_mult=1.2, sl_atr_mult=0.75,
        trail_atr_mult=1.5, cooldown_bars=5,
    ),
    "XAGUSD": dict(
        swing_lb=8, impulse_atr_mult=1.2, body_ratio_min=0.45,
        cons_bars=2, cons_range_mult=0.8, sl_atr_mult=1.0,
        trail_atr_mult=1.0, cooldown_bars=3,
    ),
    "EURJPY": dict(
        swing_lb=8, impulse_atr_mult=0.8, body_ratio_min=0.45,
        cons_bars=4, cons_range_mult=1.0, sl_atr_mult=0.5,
        trail_atr_mult=0.75, cooldown_bars=3,
    ),
    "GBPAUD": dict(
        swing_lb=3, impulse_atr_mult=1.2, body_ratio_min=0.60,
        cons_bars=2, cons_range_mult=1.0, sl_atr_mult=1.0,
        trail_atr_mult=1.5, cooldown_bars=3,
    ),
    "USDJPY": dict(
        swing_lb=5, impulse_atr_mult=1.0, body_ratio_min=0.45,
        cons_bars=2, cons_range_mult=0.8, sl_atr_mult=0.75,
        trail_atr_mult=1.5, cooldown_bars=5,
    ),
    "NZDUSD": dict(
        swing_lb=8, impulse_atr_mult=1.2, body_ratio_min=0.45,
        cons_bars=2, cons_range_mult=1.0, sl_atr_mult=0.75,
        trail_atr_mult=1.0, cooldown_bars=3,
    ),
    "AUDUSD": dict(
        swing_lb=5, impulse_atr_mult=1.0, body_ratio_min=0.60,
        cons_bars=3, cons_range_mult=1.0, sl_atr_mult=1.0,
        trail_atr_mult=1.5, cooldown_bars=3,
    ),
}

SYMBOLS = list(BEST_PARAMS.keys())

# State constants
STATE_FLAT          = "FLAT"
STATE_PENDING_ENTRY = "PENDING_ENTRY"   # signal fired, waiting for next bar open
STATE_IN_POSITION   = "IN_POSITION"


# ==============================================================================
#  SECTION 1 — SYMBOL RESOLVER
#  Tries exact name first, then common suffixes, then prefix-match.
#  Returns (broker_name, digits) or (None, None).
# ==============================================================================

SUFFIX_CANDIDATES = ["", ".m", ".a", ".", "cash", "_", "pro", "#"]

def resolve_symbol(canonical: str):
    """Return the actual broker symbol name or None."""
    # 1. Exact match
    info = mt5.symbol_info(canonical)
    if info is not None:
        if not info.visible:
            mt5.symbol_select(canonical, True)
        return canonical

    # 2. Common suffixes
    for sfx in SUFFIX_CANDIDATES[1:]:
        candidate = canonical + sfx
        info = mt5.symbol_info(candidate)
        if info is not None:
            mt5.symbol_select(candidate, True)
            logger.info(f"  {canonical} → resolved as '{candidate}'")
            return candidate

    # 3. Prefix scan
    all_syms = mt5.symbols_get() or []
    for s in all_syms:
        if s.name.upper().startswith(canonical.upper()):
            mt5.symbol_select(s.name, True)
            logger.info(f"  {canonical} → resolved as '{s.name}' (prefix match)")
            return s.name

    return None


def build_symbol_map():
    """
    Build {canonical: broker_name} for all SYMBOLS.
    Symbols that can't be resolved are excluded with a warning.
    Returns the map and the list of active canonicals.
    """
    sym_map  = {}
    active   = []
    skipped  = []
    for canon in SYMBOLS:
        broker = resolve_symbol(canon)
        if broker:
            sym_map[canon] = broker
            active.append(canon)
        else:
            skipped.append(canon)
            logger.warning(f"  {canon}: NOT FOUND on broker — excluded")

    logger.info(f"Active symbols ({len(active)}): {active}")
    if skipped:
        logger.warning(f"Skipped symbols ({len(skipped)}): {skipped}")
    return sym_map, active


# ==============================================================================
#  SECTION 2 — INDICATORS  (exact parity with backtest)
# ==============================================================================

def atr_wilder(h, l, c, period=14):
    """Wilder ATR — matches backtest _atr_wilder exactly."""
    n  = len(h)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]),
                   np.abs(l[1:] - c[:-1]))
    )
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = float(tr[:period].mean())
    k = 1.0 / period
    for i in range(period, n):
        out[i] = out[i - 1] * (1.0 - k) + tr[i] * k
    return out


def expanding_percentile_rank(arr, warmup):
    """Expanding percentile rank — matches backtest _expanding_percentile_rank."""
    n   = len(arr)
    out = np.full(n, np.nan)
    hist = []
    for i in range(warmup, n):
        v = arr[i]
        if np.isnan(v):
            continue
        if hist:
            out[i] = bisect.bisect_left(hist, v) / len(hist)
        bisect.insort(hist, v)
    return out


def swing_highs_lows(h, l, lb):
    """Swing H/L detection — matches backtest _swing_highs_lows."""
    n  = len(h)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(lb, n - lb):
        if h[i] == h[i - lb: i + lb + 1].max():
            sh[i + lb] = h[i]
        if l[i] == l[i - lb: i + lb + 1].min():
            sl[i + lb] = l[i]
    return sh, sl


def h1_bias_series(df_h1, swing_lb):
    """
    Compute H1 bias (1=bull, -1=bear, 0=neutral) and return as a
    time-indexed Series — matches backtest _h1_bias_series.
    """
    h  = df_h1["high"].values.astype(np.float64)
    l  = df_h1["low"].values.astype(np.float64)
    sh, sl_arr = swing_highs_lows(h, l, swing_lb)

    n      = len(h)
    bias   = np.zeros(n, dtype=np.int8)
    sh_vals, sl_vals = [], []

    for i in range(n):
        if not np.isnan(sh[i]):
            sh_vals.append(sh[i])
        if not np.isnan(sl_arr[i]):
            sl_vals.append(sl_arr[i])
        if len(sh_vals) >= 2 and len(sl_vals) >= 2:
            hh = sh_vals[-1] > sh_vals[-2]
            hl = sl_vals[-1] > sl_vals[-2]
            lh = sh_vals[-1] < sh_vals[-2]
            ll = sl_vals[-1] < sl_vals[-2]
            if hh and hl:
                bias[i] = 1
            elif lh and ll:
                bias[i] = -1

    idx   = df_h1["time_utc"].dt.floor("h")
    return pd.Series(bias, index=idx).groupby(level=0).last()


def session_mask(sym, utc_hours):
    """Per-symbol session filter — identical to backtest."""
    s = sym.upper()
    if s.startswith("XAU"):
        hours = GOLD_SESSION_HOURS
    elif s.startswith("XAG"):
        hours = SILVER_SESSION_HOURS
    elif any(s.startswith(pfx) for pfx in INDEX_PREFIXES):
        hours = INDEX_SESSION_HOURS
    else:
        hours = FOREX_BOTH
    return np.array([hr in hours for hr in utc_hours])


# ==============================================================================
#  SECTION 3 — BUILD INDICATOR CACHE  (called each bar from last FETCH_BARS_M5)
# ==============================================================================

def build_cache(canon, broker_sym, df_m5, df_h1, params):
    """
    Build full indicator array from raw dataframes.
    Returns dict matching the backtest cache structure.
    """
    o  = df_m5["open"].values.astype(np.float64)
    h  = df_m5["high"].values.astype(np.float64)
    l  = df_m5["low"].values.astype(np.float64)
    c  = df_m5["close"].values.astype(np.float64)
    n  = len(c)

    utc_hour = df_m5["time_utc"].dt.hour.values.astype(np.int32)
    atr14    = atr_wilder(h, l, c, ATR_PERIOD)
    atr_pct  = expanding_percentile_rank(atr14, WARMUP_M5)

    body         = np.abs(c - o)
    candle_range = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        body_ratio = np.where(candle_range > 1e-10,
                              body / candle_range, np.nan)

    sess = session_mask(canon, utc_hour)

    # H1 bias aligned to M5 timestamps
    bias_ser       = h1_bias_series(df_h1, params["swing_lb"])
    m5_floor       = df_m5["time_utc"].dt.floor("h")
    h1_bias_full   = (
        bias_ser
        .reindex(bias_ser.index.union(m5_floor.drop_duplicates()))
        .ffill()
        .reindex(m5_floor)
        .fillna(0)
        .values
        .astype(np.int8)
    )

    return {
        "canon":      canon,
        "broker":     broker_sym,
        "n":          n,
        "time_utc":   df_m5["time_utc"].values,
        "o": o, "h": h, "l": l, "c": c,
        "atr14":      atr14,
        "atr_pct":    atr_pct,
        "body":       body,
        "body_ratio": body_ratio,
        "sess_mask":  sess,
        "h1_bias":    h1_bias_full,
    }


# ==============================================================================
#  SECTION 4 — SIGNAL DETECTION  (bar-close; exact parity with backtest)
# ==============================================================================

def detect_signal_last_bar(cache, params, last_signal_bar_idx):
    """
    Check only the LAST closed bar for a new signal.
    Mirrors the backtest detect_signals logic but for a single bar.

    Returns (direction, cons_low, cons_high, atr_at_signal) or
            (None, None, None, None)
    """
    i = cache["n"] - 1  # index of last closed bar

    # Minimum data check
    imp_shift = params["cons_bars"] + 1
    if i < WARMUP_M5 + imp_shift:
        return None, None, None, None

    # ATR percentile filter
    if np.isnan(cache["atr_pct"][i]) or cache["atr_pct"][i] < ATR_PCT_THRESH:
        return None, None, None, None

    # Session filter
    if not cache["sess_mask"][i]:
        return None, None, None, None

    # H1 bias required
    bias = cache["h1_bias"][i]
    if bias == 0:
        return None, None, None, None

    # Cooldown: last_signal_bar_idx is stored as absolute bar index in current array
    # We compare using position in current data array.
    # last_signal_bar_idx is stored as absolute index; convert to relative:
    # since we always use the tail of the data, i is always len-1.
    # We track bars_since_last_signal externally and pass it in.
    # (handled by caller — see process_symbol)

    # ── Impulse bar (shifted back by cons_bars+1) ──
    imp_i = i - imp_shift
    if imp_i < 0:
        return None, None, None, None

    atr_imp = cache["atr14"][imp_i]
    if np.isnan(atr_imp) or atr_imp <= 0:
        return None, None, None, None

    body_imp  = cache["body"][imp_i]
    brat_imp  = cache["body_ratio"][imp_i]
    close_imp = cache["c"][imp_i]
    open_imp  = cache["o"][imp_i]

    is_bull_imp = (
        body_imp >= params["impulse_atr_mult"] * atr_imp and
        not np.isnan(brat_imp) and
        brat_imp >= params["body_ratio_min"] and
        close_imp > open_imp
    )
    is_bear_imp = (
        body_imp >= params["impulse_atr_mult"] * atr_imp and
        not np.isnan(brat_imp) and
        brat_imp >= params["body_ratio_min"] and
        close_imp < open_imp
    )

    # ── Consolidation window (bars [i-cons_bars .. i-1]) ──
    cb   = params["cons_bars"]
    start = i - cb
    if start < 1:
        return None, None, None, None

    cons_h = cache["h"][start:i]
    cons_l = cache["l"][start:i]
    atr_seg = cache["atr14"][start:i]

    cons_range_max = float((cons_h - cons_l).max())
    cons_atr_mean  = float(atr_seg[~np.isnan(atr_seg)].mean()) if len(atr_seg) else np.nan

    if np.isnan(cons_atr_mean) or cons_atr_mean <= 0:
        return None, None, None, None

    tight_cons = cons_range_max < params["cons_range_mult"] * cons_atr_mean

    if not tight_cons:
        return None, None, None, None

    cons_low_val  = float(cons_l.min())
    cons_high_val = float(cons_h.max())

    # ── Breakout on current bar ──
    c_cur = cache["c"][i]
    breaks_up   = c_cur > cons_high_val
    breaks_down = c_cur < cons_low_val

    atr_cur = cache["atr14"][i]
    if np.isnan(atr_cur) or atr_cur <= 0:
        return None, None, None, None

    # ── Combine ──
    long_signal  = is_bull_imp and tight_cons and breaks_up   and (bias == 1)
    short_signal = is_bear_imp and tight_cons and breaks_down and (bias == -1)

    if long_signal and not short_signal:
        return "long", cons_low_val, cons_high_val, atr_cur
    if short_signal and not long_signal:
        return "short", cons_low_val, cons_high_val, atr_cur

    return None, None, None, None


# ==============================================================================
#  SECTION 5 — DATA FETCH
# ==============================================================================

def fetch_m5(broker_sym, n=FETCH_BARS_M5):
    rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_M5, 0, n + 1)
    if rates is None or len(rates) < WARMUP_M5 + 50:
        logger.warning(
            f"[{broker_sym}] M5 fetch failed or insufficient bars: "
            f"{len(rates) if rates is not None else 0}"
        )
        return None
    cols = ["time", "open", "high", "low", "close",
            "tick_volume", "spread", "real_volume"]
    df = pd.DataFrame(rates, columns=cols)[
        ["time", "open", "high", "low", "close"]
    ].copy()
    df["time_utc"] = pd.to_datetime(df["time"].astype(np.int64), unit="s")
    df["utc_hour"] = df["time_utc"].dt.hour
    # Drop the forming (current, not yet closed) bar
    return df.iloc[:-1].reset_index(drop=True)


def fetch_h1(broker_sym, n=FETCH_BARS_H1):
    rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_H1, 0, n + 1)
    if rates is None or len(rates) < 100:
        logger.warning(
            f"[{broker_sym}] H1 fetch failed: "
            f"{len(rates) if rates is not None else 0}"
        )
        return None
    cols = ["time", "open", "high", "low", "close",
            "tick_volume", "spread", "real_volume"]
    df = pd.DataFrame(rates, columns=cols)[
        ["time", "open", "high", "low", "close"]
    ].copy()
    df["time_utc"] = pd.to_datetime(df["time"].astype(np.int64), unit="s")
    return df.iloc[:-1].reset_index(drop=True)


# ==============================================================================
#  SECTION 6 — LOT SIZE CALCULATION
# ==============================================================================

def compute_lot_size(broker_sym, entry_price, sl_price, balance):
    """1% risk per trade, sized to SL distance."""
    info = mt5.symbol_info(broker_sym)
    if info is None:
        logger.error(f"[{broker_sym}] symbol_info returned None")
        return None

    sl_dist = abs(entry_price - sl_price)
    if sl_dist < 1e-9:
        logger.error(f"[{broker_sym}] SL distance is zero")
        return None

    # Value of 1 tick (pip) per lot
    tick_value_per_lot = info.trade_tick_value / info.trade_tick_size
    risk_amount        = balance * RISK_PER_TRADE

    raw_lot = risk_amount / (sl_dist * tick_value_per_lot)
    lot     = max(
        info.volume_min,
        min(
            info.volume_max,
            round(raw_lot / info.volume_step) * info.volume_step
        )
    )
    return round(lot, 2)


# ==============================================================================
#  SECTION 7 — ORDER EXECUTION
# ==============================================================================

def send_market_order(broker_sym, direction, lot, sl_price, comment):
    """Send market entry; returns (ticket, actual_entry_price) or (None, None)."""
    tick  = mt5.symbol_info_tick(broker_sym)
    if tick is None:
        logger.error(f"[{broker_sym}] Tick unavailable")
        return None, None

    price  = tick.ask if direction == "long" else tick.bid
    otype  = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       broker_sym,
        "volume":       lot,
        "type":         otype,
        "price":        price,
        "sl":           sl_price,
        "tp":           0.0,       # no TP — exit via trail / max-hold
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      comment,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = getattr(result, "retcode", None)
        msg  = getattr(result, "comment", "")
        logger.error(f"[{broker_sym}] Entry FAILED retcode={code} msg={msg}")
        return None, None

    logger.info(
        f"[{broker_sym}] ENTRY {direction.upper()} "
        f"lot={lot} price={price:.5f} sl={sl_price:.5f} "
        f"ticket={result.order}"
    )
    return result.order, price


def modify_sl(broker_sym, ticket, new_sl, position_type):
    """Modify SL on an open position. No TP."""
    req = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   broker_sym,
        "position": ticket,
        "sl":       new_sl,
        "tp":       0.0,
    }
    result = mt5.order_send(req)
    if result is None or result.retcode not in (
        mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_NO_CHANGES
    ):
        code = getattr(result, "retcode", None)
        logger.warning(f"[{broker_sym}] SL modify failed retcode={code} new_sl={new_sl:.5f}")
        return False
    return True


def send_close_order(broker_sym, position):
    """Close an open position at market."""
    otype = (
        mt5.ORDER_TYPE_SELL
        if position.type == mt5.ORDER_TYPE_BUY
        else mt5.ORDER_TYPE_BUY
    )
    tick  = mt5.symbol_info_tick(broker_sym)
    price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       broker_sym,
        "volume":       position.volume,
        "type":         otype,
        "position":     position.ticket,
        "price":        price,
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      "msb_exit",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = getattr(result, "retcode", None)
        logger.error(f"[{broker_sym}] Close FAILED retcode={code}")
        return False
    logger.info(f"[{broker_sym}] CLOSED ticket={position.ticket} price={price:.5f}")
    return True


# ==============================================================================
#  SECTION 8 — STATE MACHINE PER SYMBOL
# ==============================================================================

def make_symbol_state():
    return {
        # State machine
        "state":           STATE_FLAT,
        # Pending entry fields (signal fired on bar close, entry on next open)
        "pending_dir":     None,
        "pending_sl":      None,
        "pending_cons_low":  None,
        "pending_cons_high": None,
        "pending_atr":     None,
        # In-position fields
        "ticket":          None,
        "direction":       None,
        "entry_price":     None,
        "sl_price":        None,
        "sl_dist":         None,
        "be_active":       False,
        "current_sl":      None,
        "hold_count":      0,
        # Cooldown tracking (bars since last signal/exit)
        "bars_since_last": 9999,
    }


def reconstruct_position_state(canon, broker_sym, position):
    """Recover state from an open MT5 position on restart."""
    entry_time = datetime.datetime.fromtimestamp(
        position.time, tz=datetime.timezone.utc
    )
    now_utc    = datetime.datetime.now(tz=datetime.timezone.utc)
    hold_count = max(0, int((now_utc - entry_time).total_seconds() / 300))  # M5 bars

    direction = "long" if position.type == mt5.ORDER_TYPE_BUY else "short"
    ep        = position.price_open
    sl_price  = position.sl
    sl_dist   = abs(ep - sl_price) if sl_price and sl_price > 0 else 0.01
    be_active = False

    # If SL has been moved to or beyond entry, assume BE is active
    if sl_price and sl_dist > 0:
        if direction == "long"  and sl_price >= ep:
            be_active = True
        if direction == "short" and sl_price <= ep:
            be_active = True

    logger.info(
        f"[{canon}] RECOVERED position: dir={direction} ep={ep:.5f} "
        f"sl={sl_price:.5f} hold≈{hold_count}bars be={be_active}"
    )
    return {
        "state":           STATE_IN_POSITION,
        "pending_dir":     None,
        "pending_sl":      None,
        "pending_cons_low":  None,
        "pending_cons_high": None,
        "pending_atr":     None,
        "ticket":          position.ticket,
        "direction":       direction,
        "entry_price":     ep,
        "sl_price":        sl_price,
        "sl_dist":         sl_dist,
        "be_active":       be_active,
        "current_sl":      sl_price,
        "hold_count":      hold_count,
        "bars_since_last": 0,
    }


# ==============================================================================
#  SECTION 9 — PER-BAR PROCESSING  (core logic)
# ==============================================================================

def process_symbol(canon, broker_sym, sym_st, params, balance):
    """
    Called once per M5 bar close for each active symbol.
    Implements the full backtest state machine in live form.

    PARITY NOTES:
      - Signal detected on bar close; entry placed at next bar open (market order)
      - SL = cons_edge ± sl_atr_mult*ATR; min dist = 5% ATR
      - BE triggered when price reaches 1R from entry
      - Trail starts after BE: SL = max(current_sl, bar_high - trail_atr_mult*ATR)
        for longs, min(current_sl, bar_low + trail_atr_mult*ATR) for shorts
      - Max hold = 60 bars; closed at market on bar 60
    """
    # ── 1. Fetch data ──────────────────────────────────────────────────────
    df_m5 = fetch_m5(broker_sym)
    df_h1 = fetch_h1(broker_sym)
    if df_m5 is None or df_h1 is None:
        logger.warning(f"[{canon}] Data fetch failed — skipping bar")
        return

    # ── 2. Build indicators ────────────────────────────────────────────────
    cache = build_cache(canon, broker_sym, df_m5, df_h1, params)

    # ── 3. Cooldown tick ───────────────────────────────────────────────────
    sym_st["bars_since_last"] += 1

    # ── 4. Broker position cross-check ────────────────────────────────────
    positions = mt5.positions_get(symbol=broker_sym) or []
    positions = [p for p in positions if p.magic == MAGIC]

    broker_has_pos = len(positions) > 0
    state_in_pos   = sym_st["state"] == STATE_IN_POSITION

    # Desync: broker has position but we think FLAT → recover
    if broker_has_pos and not state_in_pos:
        logger.warning(f"[{canon}] Desync: broker has position, state={sym_st['state']} — recovering")
        sym_st.update(reconstruct_position_state(canon, broker_sym, positions[0]))
        state_in_pos = True

    # Desync: broker closed position (SL/TP hit) but state still IN_POSITION
    if not broker_has_pos and state_in_pos:
        logger.info(f"[{canon}] Position closed server-side (SL hit or TP)")
        _log_trade_close_from_history(canon, sym_st)
        sym_st.update(make_symbol_state())
        sym_st["bars_since_last"] = 0
        return  # skip rest of processing this bar

    # ── 5. PENDING_ENTRY: place entry now (we are at the NEXT bar's open) ─
    if sym_st["state"] == STATE_PENDING_ENTRY:
        _execute_pending_entry(canon, broker_sym, sym_st, params, balance, cache)
        return  # rest of logic runs next bar

    # ── 6. IN_POSITION: manage trail and max-hold ──────────────────────────
    if sym_st["state"] == STATE_IN_POSITION:
        _manage_position(canon, broker_sym, sym_st, params, cache, positions)
        return

    # ── 7. FLAT: check for new signal ─────────────────────────────────────
    if sym_st["state"] == STATE_FLAT:
        cooldown = params["cooldown_bars"]
        if sym_st["bars_since_last"] < cooldown:
            return

        direction, cons_low, cons_high, atr_sig = detect_signal_last_bar(
            cache, params, None
        )
        if direction is not None:
            # Compute SL at signal bar (index i = len-1 of cache)
            i = cache["n"] - 1
            sl_atr = cache["atr14"][i]
            if np.isnan(sl_atr) or sl_atr <= 0:
                logger.warning(f"[{canon}] Signal skipped: ATR invalid at signal bar")
                return

            if direction == "long":
                raw_sl = cons_low - params["sl_atr_mult"] * sl_atr
            else:
                raw_sl = cons_high + params["sl_atr_mult"] * sl_atr

            logger.info(
                f"[{canon}] SIGNAL {direction.upper()} "
                f"cons_low={cons_low:.5f} cons_high={cons_high:.5f} "
                f"pending_sl={raw_sl:.5f} atr={sl_atr:.5f} "
                f"— will enter on NEXT bar open"
            )
            sym_st["state"]              = STATE_PENDING_ENTRY
            sym_st["pending_dir"]        = direction
            sym_st["pending_sl"]         = raw_sl
            sym_st["pending_cons_low"]   = cons_low
            sym_st["pending_cons_high"]  = cons_high
            sym_st["pending_atr"]        = sl_atr


def _execute_pending_entry(canon, broker_sym, sym_st, params, balance, cache):
    """
    We are at the open of the bar AFTER the signal bar.
    Place market order using current ask/bid (= next bar open approximation).
    This matches the backtest: entry at o[sig_i + 1].
    """
    direction = sym_st["pending_dir"]
    raw_sl    = sym_st["pending_sl"]
    atr_e     = sym_st["pending_atr"]

    # Get live price (next bar open)
    tick = mt5.symbol_info_tick(broker_sym)
    if tick is None:
        logger.error(f"[{canon}] Tick unavailable for entry — cancelling pending")
        sym_st.update(make_symbol_state())
        return

    ep = tick.ask if direction == "long" else tick.bid

    # Enforce minimum SL distance (5% ATR — matches backtest)
    sl_dist = abs(ep - raw_sl)
    min_sl_dist = 0.05 * atr_e
    if sl_dist < min_sl_dist:
        sl_dist = min_sl_dist
        raw_sl  = ep - sl_dist if direction == "long" else ep + sl_dist

    # Validate SL direction
    if direction == "long"  and raw_sl >= ep:
        raw_sl = ep - min_sl_dist
        sl_dist = min_sl_dist
    if direction == "short" and raw_sl <= ep:
        raw_sl = ep + min_sl_dist
        sl_dist = min_sl_dist

    lot = compute_lot_size(broker_sym, ep, raw_sl, balance)
    if lot is None:
        logger.error(f"[{canon}] Lot calc failed — cancelling pending")
        sym_st.update(make_symbol_state())
        return

    comment = f"{COMMENT}_{canon}"
    ticket, actual_ep = send_market_order(broker_sym, direction, lot, raw_sl, comment)

    if ticket is None:
        logger.error(f"[{canon}] Market order failed — cancelling pending")
        sym_st.update(make_symbol_state())
        return

    # Retrieve actual fill from broker
    time.sleep(0.5)
    positions = mt5.positions_get(symbol=broker_sym) or []
    positions = [p for p in positions if p.magic == MAGIC and p.ticket == ticket]
    if positions:
        pos       = positions[0]
        actual_ep = pos.price_open
        actual_sl = pos.sl
    else:
        actual_sl = raw_sl

    sl_dist = abs(actual_ep - actual_sl)
    if sl_dist < 1e-9:
        sl_dist = min_sl_dist

    sym_st["state"]        = STATE_IN_POSITION
    sym_st["ticket"]       = ticket
    sym_st["direction"]    = direction
    sym_st["entry_price"]  = actual_ep
    sym_st["sl_price"]     = actual_sl
    sym_st["sl_dist"]      = sl_dist
    sym_st["be_active"]    = False
    sym_st["current_sl"]   = actual_sl
    sym_st["hold_count"]   = 0
    # Clear pending fields
    sym_st["pending_dir"]  = None
    sym_st["pending_sl"]   = None
    sym_st["pending_cons_low"]  = None
    sym_st["pending_cons_high"] = None
    sym_st["pending_atr"]       = None

    logger.info(
        f"[{canon}] ENTERED {direction.upper()} "
        f"ep={actual_ep:.5f} sl={actual_sl:.5f} sl_dist={sl_dist:.5f} "
        f"lot={lot} ticket={ticket}"
    )


def _manage_position(canon, broker_sym, sym_st, params, cache, positions):
    """
    Manage an open position each bar:
      1. Increment hold count
      2. Check max-hold → close at market
      3. Check BE trigger (price reaches 1R)
      4. Update trailing stop (after BE)
      5. Modify SL on broker if changed

    Exact parity with backtest run_backtest inner loop.
    """
    i         = cache["n"] - 1
    direction = sym_st["direction"]
    ep        = sym_st["entry_price"]
    sl_dist   = sym_st["sl_dist"]
    atr_cur   = cache["atr14"][i]

    bar_h  = cache["h"][i]
    bar_l  = cache["l"][i]

    sym_st["hold_count"] += 1
    hc = sym_st["hold_count"]

    pos = positions[0] if positions else None

    # ── Max hold ──────────────────────────────────────────────────────────
    if hc >= MAX_HOLD:
        logger.info(f"[{canon}] MAX HOLD ({MAX_HOLD} bars) — closing at market")
        if pos:
            send_close_order(broker_sym, pos)
        sym_st.update(make_symbol_state())
        sym_st["bars_since_last"] = 0
        return

    # ── BE trigger ────────────────────────────────────────────────────────
    one_r = ep + (sl_dist if direction == "long" else -sl_dist)

    if not sym_st["be_active"]:
        triggered = (
            (direction == "long"  and bar_h >= one_r) or
            (direction == "short" and bar_l <= one_r)
        )
        if triggered:
            sym_st["be_active"]  = True
            sym_st["current_sl"] = ep
            logger.info(
                f"[{canon}] BREAK-EVEN triggered — SL moved to {ep:.5f}"
            )

    # ── Trail (after BE) ──────────────────────────────────────────────────
    if sym_st["be_active"]:
        ta = atr_cur if not np.isnan(atr_cur) else (sl_dist / 1.0)
        trail_mult = params["trail_atr_mult"]
        if direction == "long":
            new_sl = bar_h - trail_mult * ta
            sym_st["current_sl"] = max(sym_st["current_sl"], new_sl)
        else:
            new_sl = bar_l + trail_mult * ta
            sym_st["current_sl"] = min(sym_st["current_sl"], new_sl)

    # ── Update broker SL if it has moved ──────────────────────────────────
    if pos is not None:
        broker_sl = pos.sl or 0.0
        new_sl    = sym_st["current_sl"]
        # Only modify if change is meaningful (1 pip tolerance)
        sym_info = mt5.symbol_info(broker_sym)
        pip      = 10 ** (-sym_info.digits + 1) if sym_info else 0.0001
        if abs(new_sl - broker_sl) >= pip:
            modified = modify_sl(broker_sym, sym_st["ticket"], new_sl, pos.type)
            if modified:
                logger.info(
                    f"[{canon}] SL updated: {broker_sl:.5f} → {new_sl:.5f} "
                    f"(hold={hc} be={sym_st['be_active']})"
                )
                sym_st["sl_price"] = new_sl


def _log_trade_close_from_history(canon, sym_st):
    """Log approximate R outcome when position was closed server-side."""
    ticket = sym_st.get("ticket")
    if not ticket:
        return
    deals = mt5.history_deals_get(position=ticket)
    if deals and len(deals) >= 2:
        ep      = sym_st.get("entry_price", 0)
        sl_dist = sym_st.get("sl_dist", 1)
        close_p = deals[-1].price
        sign    = 1 if sym_st.get("direction") == "long" else -1
        r       = sign * (close_p - ep) / sl_dist if sl_dist > 0 else 0.0
        logger.info(f"[{canon}] CLOSED (server-side) price={close_p:.5f} R={r:+.3f}")
    else:
        logger.info(f"[{canon}] CLOSED (server-side) — deal history unavailable")


# ==============================================================================
#  SECTION 10 — BAR CLOCK
# ==============================================================================

_CLOCK_SYM_BROKER = None  # set in main after symbol resolution

def get_last_closed_bar_time():
    if _CLOCK_SYM_BROKER is None:
        return None
    rates = mt5.copy_rates_from_pos(_CLOCK_SYM_BROKER, mt5.TIMEFRAME_M5, 0, 2)
    if rates is not None and len(rates) >= 2:
        return pd.Timestamp(rates[1]["time"], unit="s")
    return None


def wait_for_new_bar(last_bar_time):
    while True:
        t = get_last_closed_bar_time()
        if t is not None and t > last_bar_time:
            return t
        time.sleep(10)


# ==============================================================================
#  SECTION 11 — METRICS
# ==============================================================================

class Metrics:
    def __init__(self, active_symbols):
        self.stats   = {s: {"trades": 0, "wins": 0, "total_r": 0.0}
                        for s in active_symbols}
        self.peak    = None
        self.max_dd  = 0.0
        self.last_h  = None

    def record(self, canon, r, balance):
        if canon in self.stats:
            d = self.stats[canon]
            d["trades"] += 1
            d["wins"]   += 1 if r > 0 else 0
            d["total_r"] += r
        if self.peak is None or balance > self.peak:
            self.peak = balance
        if self.peak and self.peak > 0:
            self.max_dd = max(self.max_dd, (self.peak - balance) / self.peak)

    def report(self, balance):
        tot_t = sum(d["trades"] for d in self.stats.values())
        tot_w = sum(d["wins"]   for d in self.stats.values())
        tot_r = sum(d["total_r"] for d in self.stats.values())
        wr    = tot_w / tot_t if tot_t else 0.0
        exp   = tot_r / tot_t if tot_t else 0.0
        logger.info(
            f"\n{'='*70}\n[HOURLY REPORT — MSB LIVE]\n"
            f"  Total trades  : {tot_t}\n"
            f"  Win rate      : {wr:.1%}\n"
            f"  Expectancy    : {exp:+.3f}R\n"
            f"  Total R       : {tot_r:+.1f}\n"
            f"  Max DD        : {self.max_dd:.1%}\n"
            f"  Equity        : {balance:,.2f}\n"
        )
        for canon, d in sorted(self.stats.items(), key=lambda x: -x[1]["total_r"]):
            if d["trades"] > 0:
                swr  = d["wins"] / d["trades"]
                sexp = d["total_r"] / d["trades"]
                logger.info(
                    f"  {canon:<8}  n={d['trades']:>4}  "
                    f"WR={swr:.1%}  E={sexp:+.3f}R  totalR={d['total_r']:+.1f}"
                )
        logger.info("=" * 70)

    def check_hourly(self, balance):
        h = datetime.datetime.now(datetime.timezone.utc).hour
        if self.last_h is None:
            self.last_h = h
        if h != self.last_h:
            self.last_h = h
            self.report(balance)


# ==============================================================================
#  SECTION 12 — MAIN LOOP
# ==============================================================================

def run_live():
    global _CLOCK_SYM_BROKER

    if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN,
                          password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    acct = mt5.account_info()
    logger.info(f"MT5 connected | account={acct.login} | balance={acct.balance:.2f}")
    logger.info(f"Engine: MSB LIVE | Magic: {MAGIC} | Comment: {COMMENT}")

    # ── Symbol resolution and diagnostic ─────────────────────────────────
    logger.info("=== SYMBOL DIAGNOSTIC ===")
    sym_map, active_symbols = build_symbol_map()

    for canon in active_symbols:
        broker = sym_map[canon]
        info   = mt5.symbol_info(broker)
        tick   = mt5.symbol_info_tick(broker)
        bars_t = mt5.copy_rates_from_pos(broker, mt5.TIMEFRAME_M5, 0, 5)
        logger.info(
            f"  {canon} ({broker}): "
            f"visible={info.visible}  spread={info.spread}  "
            f"digits={info.digits}  "
            f"tick={'OK' if tick else 'None'}  "
            f"m5_bars={'OK len='+str(len(bars_t)) if bars_t is not None else 'None'}"
        )
    logger.info("=== END DIAGNOSTIC ===")

    if not active_symbols:
        logger.error("No symbols available — shutting down")
        mt5.shutdown()
        return

    # ── Set bar clock reference ───────────────────────────────────────────
    _CLOCK_SYM_BROKER = sym_map[active_symbols[0]]
    logger.info(f"Bar clock reference: {active_symbols[0]} ({_CLOCK_SYM_BROKER})")

    # ── Per-symbol state machines ─────────────────────────────────────────
    sym_states = {canon: make_symbol_state() for canon in active_symbols}

    # ── Recover any open positions on startup ─────────────────────────────
    logger.info("=== STARTUP RECOVERY ===")
    for canon in active_symbols:
        broker = sym_map[canon]
        positions = mt5.positions_get(symbol=broker) or []
        positions = [p for p in positions if p.magic == MAGIC]
        if positions:
            pos = positions[0]
            comment_ok = COMMENT in (pos.comment or "")
            if comment_ok:
                sym_states[canon].update(
                    reconstruct_position_state(canon, broker, pos)
                )
            else:
                logger.warning(
                    f"  {canon}: open position ticket={pos.ticket} "
                    f"has different comment='{pos.comment}' — not adopted"
                )
        else:
            logger.info(f"  {canon}: no open position")
    logger.info("=== END RECOVERY ===")

    # ── Log params for each active symbol ─────────────────────────────────
    logger.info("=== ACTIVE PARAMS ===")
    for canon in active_symbols:
        p = BEST_PARAMS[canon]
        logger.info(f"  {canon}: {p}")
    logger.info(f"  RISK_PER_TRADE={RISK_PER_TRADE:.2%}  MAX_HOLD={MAX_HOLD}bars  "
                f"ATR_PCT_THRESH={ATR_PCT_THRESH}  WARMUP={WARMUP_M5}bars")
    logger.info("=== END PARAMS ===")

    metrics   = Metrics(active_symbols)
    bar_count = 0

    # ── Seed bar clock ────────────────────────────────────────────────────
    last_bar_time = get_last_closed_bar_time()
    if last_bar_time is None:
        last_bar_time = pd.Timestamp.utcnow()
    logger.info(f"Seeded bar time: {last_bar_time} — waiting for next M5 close...")

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        try:
            new_bar_time  = wait_for_new_bar(last_bar_time)
            last_bar_time = new_bar_time
            bar_count    += 1

            logger.info(
                f"── BAR {bar_count} | {new_bar_time} UTC "
                f"────────────────────────────────"
            )

            balance = mt5.account_info().balance

            for canon in active_symbols:
                broker = sym_map[canon]
                params = BEST_PARAMS[canon]
                try:
                    process_symbol(canon, broker, sym_states[canon], params, balance)
                except Exception as sym_err:
                    logger.exception(f"[{canon}] Error in process_symbol: {sym_err}")

            metrics.check_hourly(balance)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down MSB LIVE")
            break
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(60)

    mt5.shutdown()
    logger.info("MT5 disconnected. MSB LIVE engine stopped.")
    try:
        final_balance = mt5.account_info().balance if mt5.account_info() else 0
    except Exception:
        final_balance = 0
    metrics.report(final_balance)


# ==============================================================================
#  ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_live()
