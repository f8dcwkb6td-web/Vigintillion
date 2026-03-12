"""
==============================================================================
MULTI-MODEL FX STRATEGY ENGINE  |  M1  |  VECTORIZED  |  NO LOOKAHEAD
==============================================================================
Model 1 : London Liquidity Expansion        (medium vol regime, 07:00-10:00 UTC)
Model 2 : New York Trend Exhaustion         (high vol regime,   12:30-15:00 UTC)
Model 3 : Volatility Compression Breakout   (low vol regime,    excl rollover)

Rules
  - Signal detected at close of bar i using ONLY data[0..i]
  - Entry price  = open[i+1]  (no lookahead)
  - Spread-aware: long  entry = open[i+1] + spread[i+1]/2
                  short entry = open[i+1] - spread[i+1]/2
  - SL/TP triggers: long uses bid, short uses ask
  - Session windows use UTC timestamps
  - Grid <= 20,000 combos per model

Outputs
  mme_m{1,2,3}_trades.csv
  mme_m{1,2,3}_grid.csv
  mme_m{1,2,3}_agg.csv
  multi_model_engine.log
==============================================================================
"""

import os, sys, io, logging, itertools, time, bisect
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("MME")
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler("multi_model_engine.log", maxBytes=20_000_000,
                           backupCount=2, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
logger.addHandler(_fh)
_sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer,
                             encoding="utf-8", errors="replace"))
_sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_sh)

# ── MT5 connection ─────────────────────────────────────────────────────────────
TERMINAL_PATH  = os.environ.get("MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
LOGIN          = int(os.environ.get("MT5_LOGIN",    7376407))
PASSWORD       = os.environ.get("MT5_PASSWORD", "iC8XoiRp&4L4KU")
SERVER         = os.environ.get("MT5_SERVER",   "ICMarketsSC-MT5-2")

FETCH_BARS     = 200_000      # reduced from 500k — stays within broker history limits
SYMBOLS        = ["AUDJPY", "EURJPY", "EURAUD", "GBPJPY", "USDJPY"]
MIN_TRADES     = 50
RISK_PER_TRADE = 0.06
MAX_HOLD_BARS  = 120
FIXED_SPREAD   = 0.0002
FETCH_RETRIES  = 5
FETCH_DELAY    = 2.0          # seconds between retries

RV_LOW_PCT  = 40
RV_MED_LO   = 40
RV_MED_HI   = 60
RV_HIGH_PCT = 80


# ══════════════════════════════════════════════════════════════════════════════
#  GRID DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

GRID_GLOBAL = {
    "rv_window":            [120, 60],
    "efficiency_window":    [20,  10],
    "efficiency_threshold": [0.3, 0.2],
}

GRID_M1 = {
    "asia_range_pct":    [25, 35, 45],
    "body_mult":         [1.5, 1.0, 0.7],
    "tp_mult":           [3.0, 2.0, 1.5],
    "stop_buffer_atr":   [0.2, 0.0],
}

GRID_M2 = {
    "ldn_move_pct":      [80, 70, 60],
    "wick_mult":         [2.5, 2.0, 1.5],
    "tp_multiple":       [3.0, 2.0, 1.5],
    "stop_buffer_atr":   [0.2, 0.0],
}

GRID_M3 = {
    "box_lookback":      [60, 40, 20],
    "box_atr_ratio":     [1.5, 1.0, 0.7],
    "expansion_mult":    [1.5, 1.2, 1.0],
    "tp_mult":           [3.0, 2.0, 1.5],
}

def _ncombos(g):
    r = 1
    for v in g.values(): r *= len(v)
    return r

assert _ncombos(GRID_GLOBAL) * _ncombos(GRID_M1) <= 20_000
assert _ncombos(GRID_GLOBAL) * _ncombos(GRID_M2) <= 20_000
assert _ncombos(GRID_GLOBAL) * _ncombos(GRID_M3) <= 20_000


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS BAR
# ══════════════════════════════════════════════════════════════════════════════

class Bar:
    def __init__(self, total, prefix="", w=48):
        self.total = total; self.w = w
        self.prefix = prefix; self.cur = 0
        self.t0 = time.time(); self._show(0)

    def step(self, n=1):
        self.cur = min(self.cur + n, self.total); self._show(self.cur)

    def _show(self, done):
        p  = done / self.total if self.total else 1
        f  = int(self.w * p)
        el = time.time() - self.t0
        eta = (el / p - el) if p > 0.001 else 0
        sys.stdout.write(
            f"\r{self.prefix} [{'█'*f}{'░'*(self.w-f)}] "
            f"{done}/{self.total} {p*100:.0f}%  {el:.0f}s  ETA {eta:.0f}s   "
        )
        sys.stdout.flush()
        if done >= self.total:
            sys.stdout.write("\n"); sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  MT5 CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def connect_mt5():
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package not installed")

    logger.info("Initializing MT5 terminal...")

    ok = mt5.initialize(
        path=TERMINAL_PATH,
        login=LOGIN,
        password=PASSWORD,
        server=SERVER,
        timeout=60_000,
        portable=False,
    )
    if not ok:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"mt5.initialize failed: {err}")

    info = mt5.account_info()
    if info is None:
        mt5.shutdown()
        raise RuntimeError(f"account_info returned None after init: {mt5.last_error()}")

    logger.info(f"MT5 connected | Account {info.login} | Server {info.server} | Balance {info.balance}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCH  — robust retry, correct symbol resolution, broker-time → UTC
# ══════════════════════════════════════════════════════════════════════════════

def resolve_symbol(base_name):
    """
    IC Markets often appends suffixes like 'm', '.r', '.pro', etc.
    Try the bare name first, then search for the best match.
    """
    # Try exact name first
    info = mt5.symbol_info(base_name)
    if info is not None:
        return base_name

    # Search for any symbol containing the base name
    all_syms = mt5.symbols_get()
    if all_syms is None:
        return None

    candidates = [s.name for s in all_syms if base_name in s.name]
    if not candidates:
        return None

    # Prefer shortest match (closest to the bare name)
    candidates.sort(key=len)
    logger.info(f"  {base_name}: resolved to '{candidates[0]}' (candidates: {candidates[:5]})")
    return candidates[0]


def fetch(base_sym, n=FETCH_BARS):
    # Resolve broker-specific symbol name
    sym = resolve_symbol(base_sym)
    if sym is None:
        logger.warning(f"  {base_sym}: no matching symbol found on broker")
        return None

    # Select and warm up the symbol feed
    if not mt5.symbol_select(sym, True):
        logger.warning(f"  {sym}: symbol_select failed — {mt5.last_error()}")
        return None

    # Give the terminal time to subscribe and populate history
    time.sleep(1.0)

    rates = None
    for attempt in range(1, FETCH_RETRIES + 1):
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, n)
        if rates is not None and len(rates) >= 2000:
            break
        err = mt5.last_error()
        logger.warning(f"  {sym}: attempt {attempt}/{FETCH_RETRIES} failed — {err}")
        if attempt < FETCH_RETRIES:
            time.sleep(FETCH_DELAY)

    if rates is None or len(rates) < 2000:
        logger.warning(f"  {sym}: all fetch attempts failed")
        return None

    df = pd.DataFrame(rates)

    # ── Broker time → UTC ─────────────────────────────────────────────────────
    # IC Markets uses EET/EEST (Europe/Athens): UTC+2 winter, UTC+3 summer.
    # MT5 timestamps are Unix seconds but represent broker wall-clock time.
    raw_ts = pd.to_datetime(df["time"], unit="s")
    try:
        broker_aware = raw_ts.dt.tz_localize(
            "Europe/Athens", ambiguous="infer", nonexistent="shift_forward"
        )
        utc_ts = broker_aware.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        logger.warning(f"  {sym}: DST conversion failed — using fixed UTC-2 offset")
        utc_ts = raw_ts - pd.Timedelta(hours=2)

    df["time"]       = utc_ts
    df["h_utc"]      = df["time"].dt.hour
    df["m_utc"]      = df["time"].dt.minute
    df["date"]       = df["time"].dt.date
    df["min_of_day"] = df["h_utc"] * 60 + df["m_utc"]

    # Spread in price terms
    if "spread" in df.columns:
        info  = mt5.symbol_info(sym)
        point = info.point if info and info.point > 0 else 0.0001
        df["spread_price"] = df["spread"].astype(float) * point
    else:
        df["spread_price"] = np.nan

    logger.info(
        f"  {sym}: {len(df):,} bars  "
        f"{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}  "
        f"spread={'col' if 'spread' in df.columns else 'fixed'}"
    )
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BASE CACHE — computed once per symbol, no grid params
# ══════════════════════════════════════════════════════════════════════════════

def build_base_cache(df):
    o   = df["open"].values.astype(np.float64)
    h   = df["high"].values.astype(np.float64)
    l   = df["low"].values.astype(np.float64)
    c   = df["close"].values.astype(np.float64)
    sp  = df["spread_price"].values.astype(np.float64)
    mod = df["min_of_day"].values.astype(np.int32)
    dt  = df["date"].values
    n   = len(c)

    # ATR-14 (EWM)
    tr    = np.maximum(h - l,
            np.maximum(np.abs(h - np.roll(c, 1)),
                       np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr   = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    body      = np.abs(c - o)
    bar_range = h - l
    med_body  = pd.Series(body).rolling(100, min_periods=10).median().values
    med_range = pd.Series(bar_range).rolling(100, min_periods=10).median().values

    # Session masks (UTC minute-of-day)
    asia_mask   = (mod >= 0)    & (mod < 360)    # 00:00–06:00
    london_mask = (mod >= 420)  & (mod < 600)    # 07:00–10:00
    ny_mask     = (mod >= 750)  & (mod < 900)    # 12:30–15:00
    rollover    = (mod >= 1380) | (mod < 60)     # 23:00–01:00

    # ── Asian range per day (built forward, no lookahead)
    unique_d = np.unique(dt)
    day_ah, day_al, day_ar = {}, {}, {}
    for d in unique_d:
        m = (dt == d) & asia_mask
        if m.any():
            day_ah[d] = h[m].max()
            day_al[d] = l[m].min()
            day_ar[d] = day_ah[d] - day_al[d]

    a_hi  = np.full(n, np.nan)
    a_lo  = np.full(n, np.nan)
    a_rng = np.full(n, np.nan)
    a_pct = np.full(n, np.nan)
    sorted_ar = []; seen_ad = set()
    for i in range(n):
        if london_mask[i]:
            d = dt[i]
            if d in day_ah:
                a_hi[i]  = day_ah[d]
                a_lo[i]  = day_al[d]
                a_rng[i] = day_ar[d]
                r = day_ar[d]
                if sorted_ar:
                    a_pct[i] = bisect.bisect_left(sorted_ar, r) / len(sorted_ar) * 100
                if d not in seen_ad:
                    bisect.insort(sorted_ar, r)
                    seen_ad.add(d)

    # ── London move per day
    ldn_o_bar = (mod >= 420) & (mod < 421)
    ldn_c_bar = (mod >= 719) & (mod < 720)
    lo_by_d, lc_by_d = {}, {}
    for i in range(n):
        d = dt[i]
        if ldn_o_bar[i]: lo_by_d[d] = o[i]
        if ldn_c_bar[i]: lc_by_d[d] = c[i]

    ldn_move = np.full(n, np.nan)
    ldn_mpct = np.full(n, np.nan)
    sorted_lm = []; seen_ld = set()
    for i in range(n):
        if ny_mask[i]:
            d = dt[i]
            if d in lo_by_d and d in lc_by_d:
                mv = lc_by_d[d] - lo_by_d[d]
                ldn_move[i] = mv
                ab = abs(mv)
                if sorted_lm:
                    ldn_mpct[i] = bisect.bisect_left(sorted_lm, ab) / len(sorted_lm) * 100
                if d not in seen_ld:
                    bisect.insort(sorted_lm, ab)
                    seen_ld.add(d)

    log_ret = np.diff(np.log(np.maximum(c, 1e-9)), prepend=np.log(c[0]))

    return {
        "n": n, "o": o, "h": h, "l": l, "c": c, "sp": sp,
        "atr": atr, "body": body, "bar_range": bar_range,
        "med_body": med_body, "med_range": med_range,
        "mod": mod, "dt": dt,
        "asia_mask": asia_mask, "london_mask": london_mask,
        "ny_mask": ny_mask, "rollover": rollover,
        "a_hi": a_hi, "a_lo": a_lo, "a_rng": a_rng, "a_pct": a_pct,
        "ldn_move": ldn_move, "ldn_mpct": ldn_mpct,
        "log_ret": log_ret,
        "times": df["time"].values,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PARAM-DEPENDENT RECOMPUTE
# ══════════════════════════════════════════════════════════════════════════════

def compute_regimes_and_eff(base, rv_w, eff_w):
    n  = base["n"]
    c  = base["c"]
    lr = base["log_ret"]

    rv_raw = np.sqrt(np.maximum(
        pd.Series(lr ** 2).rolling(rv_w, min_periods=rv_w // 2).sum().values, 0.0))

    rv_pct    = np.full(n, 50.0)
    sorted_rv = []
    warmup    = max(rv_w * 3, 500)
    for i in range(n):
        if i >= warmup and sorted_rv:
            rv_pct[i] = bisect.bisect_left(sorted_rv, rv_raw[i]) / len(sorted_rv) * 100
        if i % 5 == 0:
            bisect.insort(sorted_rv, rv_raw[i])

    rl = rv_pct < RV_LOW_PCT
    rm = (rv_pct >= RV_MED_LO) & (rv_pct <= RV_MED_HI)
    rh = rv_pct > RV_HIGH_PCT

    shifted = pd.Series(c).shift(eff_w).values
    net     = np.abs(c - shifted)
    gross   = pd.Series(
        np.abs(np.diff(c, prepend=c[0]))
    ).rolling(eff_w, min_periods=2).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(gross > 0, net / gross, 0.0)

    return rl, rm, rh, eff


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

def detect_m1(base, regime_med, eff, eff_thr, pm):
    """London Liquidity Expansion"""
    c        = base["c"]; o = base["o"]
    atr      = base["atr"]
    body     = base["body"]; med_body = base["med_body"]
    a_hi     = base["a_hi"]; a_lo = base["a_lo"]
    a_rng    = base["a_rng"]; a_pct = base["a_pct"]
    win      = base["london_mask"]

    valid = (win
             & regime_med
             & (eff >= eff_thr)
             & ~np.isnan(a_hi)
             & ~np.isnan(a_pct)
             & (a_pct < pm["asia_range_pct"])
             & (a_rng > 0)
             & (body >= pm["body_mult"] * med_body))

    long_s  = valid & (c > a_hi)
    short_s = valid & (c < a_lo)

    n    = base["n"]
    dirs = np.zeros(n, dtype=np.int8)
    dirs[long_s]  =  1
    dirs[short_s] = -1

    buf = pm["stop_buffer_atr"]
    sl  = np.full(n, np.nan)
    sl[long_s]  = a_lo[long_s]  - buf * atr[long_s]
    sl[short_s] = a_hi[short_s] + buf * atr[short_s]

    tp_dist = a_rng * pm["tp_mult"]
    return dirs, sl, ("fixed_dist", tp_dist)


def detect_m2(base, regime_high, eff, eff_thr, pm):
    """NY Trend Exhaustion Reversal"""
    c = base["c"]; o = base["o"]; h = base["h"]; l = base["l"]
    atr      = base["atr"]
    body     = base["body"]
    ldn_move = base["ldn_move"]
    ldn_mpct = base["ldn_mpct"]
    win      = base["ny_mask"]

    has_move    = ~np.isnan(ldn_move) & ~np.isnan(ldn_mpct)
    strong_move = has_move & (ldn_mpct >= pm["ldn_move_pct"])
    bull_london = strong_move & (ldn_move > 0)
    bear_london = strong_move & (ldn_move < 0)

    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    safe_body  = np.maximum(body, 1e-9)

    bear_rej = (bull_london & (c < o) & (upper_wick >= pm["wick_mult"] * safe_body))
    bull_rej = (bear_london & (c > o) & (lower_wick >= pm["wick_mult"] * safe_body))

    valid_s = win & regime_high & (eff >= eff_thr) & bear_rej
    valid_l = win & regime_high & (eff >= eff_thr) & bull_rej

    n    = base["n"]
    dirs = np.zeros(n, dtype=np.int8)
    dirs[valid_l] =  1
    dirs[valid_s] = -1

    buf = pm["stop_buffer_atr"]
    sl  = np.full(n, np.nan)
    sl[valid_l] = l[valid_l] - buf * atr[valid_l]
    sl[valid_s] = h[valid_s] + buf * atr[valid_s]

    return dirs, sl, ("sl_mult", pm["tp_multiple"])


def detect_m3(base, regime_low, eff, eff_thr, pm):
    """Volatility Compression Breakout"""
    c = base["c"]; h = base["h"]; l = base["l"]
    atr       = base["atr"]
    bar_range = base["bar_range"]
    med_range = base["med_range"]
    rollover  = base["rollover"]

    lkbk  = pm["box_lookback"]
    box_h = pd.Series(h).rolling(lkbk, min_periods=lkbk // 2).max().shift(1).values
    box_l = pd.Series(l).rolling(lkbk, min_periods=lkbk // 2).min().shift(1).values
    box_r = box_h - box_l

    valid_base = (~rollover
                  & regime_low
                  & (eff >= eff_thr)
                  & ~np.isnan(box_h)
                  & (box_r > 0)
                  & (box_r <= pm["box_atr_ratio"] * atr)
                  & (bar_range >= pm["expansion_mult"] * med_range))

    long_s  = valid_base & (c > box_h)
    short_s = valid_base & (c < box_l)

    n    = base["n"]
    dirs = np.zeros(n, dtype=np.int8)
    dirs[long_s]  =  1
    dirs[short_s] = -1

    box_mid = (box_h + box_l) / 2.0
    sl      = np.full(n, np.nan)
    sl[long_s]  = box_mid[long_s]
    sl[short_s] = box_mid[short_s]

    return dirs, sl, ("box_mult", box_r, pm["tp_mult"])


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZED BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def backtest_vec(base, dirs, sl_arr, tp_info, model_id,
                 regime_low, regime_med, regime_high):
    n     = base["n"]
    o     = base["o"]; h = base["h"]; l = base["l"]; c = base["c"]
    sp    = base["sp"]
    times = base["times"]

    sig_idx = np.where(dirs != 0)[0]
    sig_idx = sig_idx[sig_idx + 1 < n]
    if len(sig_idx) == 0:
        return None, []

    ei  = sig_idx + 1
    d   = dirs[sig_idx]

    sp_e = np.where(np.isnan(sp[ei]), FIXED_SPREAD, sp[ei])
    ep   = np.where(d == 1,
                    o[ei] + sp_e / 2.0,
                    o[ei] - sp_e / 2.0)

    sl_p    = sl_arr[sig_idx]
    sl_dist = np.maximum(np.abs(ep - sl_p), 1e-9)

    tp_type = tp_info[0]
    if tp_type == "fixed_dist":
        tp_d = tp_info[1][sig_idx]
        tp_p = np.where(d == 1, ep + tp_d, ep - tp_d)
    elif tp_type == "sl_mult":
        mult = tp_info[1]
        tp_p = np.where(d == 1, ep + sl_dist * mult, ep - sl_dist * mult)
    else:  # "box_mult"
        br   = tp_info[1][sig_idx]; mult = tp_info[2]
        tp_p = np.where(d == 1, ep + br * mult, ep - br * mult)

    nt      = len(sig_idx)
    max_cap = MAX_HOLD_BARS
    offsets = np.arange(max_cap)
    abs_i   = np.clip(ei[:, None] + offsets[None, :], 0, n - 1)

    fut_h  = h[abs_i]; fut_l = l[abs_i]; fut_c = c[abs_i]
    sp_fwd = np.where(np.isnan(sp[abs_i]), FIXED_SPREAD, sp[abs_i])

    off_mask = offsets[None, :] < max_cap

    sl_hit = np.where(d[:, None] == 1,
                      fut_l <= sl_p[:, None],
                      fut_h + sp_fwd >= sl_p[:, None]) & off_mask
    tp_hit = np.where(d[:, None] == 1,
                      fut_h >= tp_p[:, None],
                      fut_l + sp_fwd <= tp_p[:, None]) & off_mask

    any_exit = sl_hit | tp_hit
    first    = np.argmax(any_exit, axis=1)
    hit      = any_exit[np.arange(nt), first]
    exit_off = np.where(hit, first, max_cap - 1)

    at_sl  = sl_hit[np.arange(nt), exit_off]
    at_tp  = tp_hit[np.arange(nt), exit_off]
    exit_c = fut_c[np.arange(nt), exit_off]

    raw_r    = np.clip((exit_c - ep) / sl_dist * d, -2.0, 4.0)
    tp_rr    = np.abs(tp_p - ep) / sl_dist
    result_r = np.where(at_sl, -1.0, np.where(at_tp, tp_rr, raw_r))

    reg = np.where(regime_low[sig_idx],  "low",
          np.where(regime_med[sig_idx],  "med",
          np.where(regime_high[sig_idx], "high", "neutral")))

    trades = []
    for k in range(nt):
        trades.append({
            "timestamp":         str(times[ei[k]]),
            "model_id":          model_id,
            "direction":         "long" if d[k] == 1 else "short",
            "entry_price":       round(float(ep[k]),    6),
            "stop_price":        round(float(sl_p[k]),  6),
            "target_price":      round(float(tp_p[k]),  6),
            "spread_used":       round(float(sp_e[k]),  6),
            "volatility_regime": str(reg[k]),
            "result_r":          round(float(result_r[k]), 4),
        })

    if len(trades) < MIN_TRADES:
        return None, trades

    rr   = result_r
    wins = rr > 0
    tt   = len(rr)
    wr   = wins.sum() / tt
    wr_  = float(rr[wins].sum())
    lr_  = float(np.abs(rr[~wins].sum())) if (~wins).sum() > 0 else 1.0
    nw   = wins.sum(); nl = (~wins).sum()
    exp  = wr * (wr_ / nw if nw else 0) - (1 - wr) * (lr_ / nl if nl else 1)
    pf   = wr_ / lr_ if lr_ > 0 else 999.0

    bal  = 10_000.0; bals = [bal]
    for r in rr:
        bal = bal * (1 + RISK_PER_TRADE * r) if r > 0 \
              else bal * (1 - RISK_PER_TRADE * abs(r))
        bals.append(bal)
    bals = np.array(bals)
    rm_a = np.maximum.accumulate(bals)
    dd   = rm_a - bals
    mdd  = float(dd.max() / rm_a[np.argmax(dd)]) if len(bals) > 1 else 0.0

    t0  = pd.Timestamp(times[0]); t1 = pd.Timestamp(times[-1])
    wks = max((t1 - t0).total_seconds() / 604_800, 1.0)

    return {
        "total_trades":    tt,
        "win_rate":        round(wr,  4),
        "expectancy_r":    round(exp, 4),
        "profit_factor":   round(pf,  3),
        "max_drawdown":    round(mdd, 4),
        "trades_per_week": round(tt / wks, 2),
    }, trades


# ══════════════════════════════════════════════════════════════════════════════
#  GRID RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_model_grid(model_id, detect_fn, grid_m, base_caches):
    keys_g   = list(GRID_GLOBAL.keys())
    keys_m   = list(grid_m.keys())
    combos_g = list(itertools.product(*[GRID_GLOBAL[k] for k in keys_g]))
    combos_m = list(itertools.product(*[grid_m[k]      for k in keys_m]))
    total    = len(combos_g) * len(combos_m)

    all_rows   = []
    all_trades = []

    for sym, base in base_caches.items():
        pb = Bar(total, prefix=f"  M{model_id} {sym}")

        for g_vals in combos_g:
            pg   = dict(zip(keys_g, g_vals))
            rv_w = pg["rv_window"]
            ew   = pg["efficiency_window"]
            ethr = pg["efficiency_threshold"]

            rl, rm, rh, eff = compute_regimes_and_eff(base, rv_w, ew)

            for m_vals in combos_m:
                pm = dict(zip(keys_m, m_vals))

                try:
                    regime_for_model = (rm if model_id == 1
                                        else rh if model_id == 2
                                        else rl)
                    dirs, sl_arr, tp_info = detect_fn(
                        base, regime_for_model, eff, ethr, pm)
                    stats, trades = backtest_vec(
                        base, dirs, sl_arr, tp_info,
                        model_id, rl, rm, rh)
                except Exception as e:
                    logger.debug(f"  combo error: {e}")
                    stats = None; trades = []

                if stats is not None:
                    row = {**pg, **pm, "symbol": sym, **stats}
                    all_rows.append(row)
                    for t in trades:
                        t["symbol"] = sym
                    all_trades.extend(trades)

                pb.step()

    return all_rows, all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_top(agg, model_id, keys_m, top=10):
    sep = "─" * 92
    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL {model_id} — TOP {top}  (mean expectancy across all symbols)")
    logger.info(sep)
    for i, row in agg.head(top).iterrows():
        ps = "  ".join(f"{k}={row[k]}" for k in (list(GRID_GLOBAL.keys()) + keys_m))
        logger.info(
            f"  #{i+1:<3} WR={row['win_rate']:.1%}  E={row['expectancy_r']:+.3f}  "
            f"PF={row['profit_factor']:.2f}  MDD={row['max_drawdown']:.1%}  "
            f"T/wk={row['trades_per_week']:.1f}"
        )
        logger.info(f"       {ps}")
    if len(agg):
        best = agg.iloc[0]
        logger.info(f"\n  ★ BEST COMBO MODEL {model_id}")
        for k in list(GRID_GLOBAL.keys()) + keys_m:
            logger.info(f"    {k:<28}: {best[k]}")
        logger.info(f"    {'win_rate':<28}: {best['win_rate']:.1%}")
        logger.info(f"    {'expectancy_r':<28}: {best['expectancy_r']:+.4f}R")
        logger.info(f"    {'profit_factor':<28}: {best['profit_factor']:.3f}")
        logger.info(f"    {'max_drawdown':<28}: {best['max_drawdown']:.2%}")
        logger.info(f"    {'trades_per_week':<28}: {best['trades_per_week']:.1f}")
    logger.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    connect_mt5()

    logger.info("=" * 70)
    logger.info("MULTI-MODEL FX ENGINE — VECTORIZED GRID SEARCH")
    logger.info(
        f"M1 combos: {_ncombos(GRID_GLOBAL)*_ncombos(GRID_M1):,}  "
        f"M2: {_ncombos(GRID_GLOBAL)*_ncombos(GRID_M2):,}  "
        f"M3: {_ncombos(GRID_GLOBAL)*_ncombos(GRID_M3):,}"
    )
    logger.info("No lookahead: signal bar i → entry open[i+1]")
    logger.info("=" * 70)

    # ── Fetch all data before shutting MT5 down ────────────────────────────
    base_caches = {}
    for base_sym in SYMBOLS:
        logger.info(f"\n[{base_sym}] fetching...")
        df = fetch(base_sym)
        if df is None:
            logger.warning(f"[{base_sym}] skipped — no data")
            continue
        logger.info(f"[{base_sym}] building cache...")
        base_caches[base_sym] = build_base_cache(df)

    mt5.shutdown()
    logger.info("\nMT5 connection closed")

    if not base_caches:
        logger.error("No data fetched — aborting")
        return

    # ── Run models ─────────────────────────────────────────────────────────
    model_defs = [
        (1, detect_m1, GRID_M1, list(GRID_M1.keys())),
        (2, detect_m2, GRID_M2, list(GRID_M2.keys())),
        (3, detect_m3, GRID_M3, list(GRID_M3.keys())),
    ]

    for model_id, detect_fn, grid_m, keys_m in model_defs:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING MODEL {model_id}")

        rows, trades = run_model_grid(model_id, detect_fn, grid_m, base_caches)

        if not rows:
            logger.warning(f"Model {model_id}: no valid parameter combinations")
            continue

        df_full = pd.DataFrame(rows)
        df_full.to_csv(f"mme_m{model_id}_grid.csv", index=False)

        if trades:
            pd.DataFrame(trades).to_csv(f"mme_m{model_id}_trades.csv", index=False)

        agg_keys = list(GRID_GLOBAL.keys()) + keys_m
        agg = (df_full
               .groupby(agg_keys)
               .agg(
                   symbols_valid   = ("symbol",          "count"),
                   win_rate        = ("win_rate",         "mean"),
                   expectancy_r    = ("expectancy_r",     "mean"),
                   profit_factor   = ("profit_factor",    "mean"),
                   max_drawdown    = ("max_drawdown",     "mean"),
                   trades_per_week = ("trades_per_week",  "mean"),
               )
               .reset_index())

        agg = agg[agg["symbols_valid"] == len(base_caches)]
        agg = agg.sort_values("expectancy_r", ascending=False).reset_index(drop=True)
        agg.to_csv(f"mme_m{model_id}_agg.csv", index=False)

        print_top(agg, model_id, keys_m)

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
