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
  - Session windows use UTC timestamps — NOT bar counts
  - Grid <= 20,000 combos per model

Outputs
  mme_m{1,2,3}_trades.csv  — every trade with all required fields
  mme_m{1,2,3}_grid.csv    — full grid results per symbol
  mme_m{1,2,3}_agg.csv     — aggregated results across all symbols
  multi_model_engine.log   — full run log
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
TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH",
                               r"C:\Program Files\MetaTrader 5\terminal64.exe")
LOGIN    = 7376407
PASSWORD = "iC8XoiRp&4L4KU"
SERVER   = "ICMarketsSC-MT5-2"

FETCH_BARS     = 500_000
SYMBOLS        = ["AUDJPY", "EURJPY", "EURAUD", "GBPJPY", "USDJPY"]
MIN_TRADES     = 50
RISK_PER_TRADE = 0.06
MAX_HOLD_BARS  = 120      # 2-hour hard cap

# Fixed vol regime thresholds (per spec)
RV_LOW_PCT  = 40
RV_MED_LO   = 40
RV_MED_HI   = 60
RV_HIGH_PCT = 80


# ══════════════════════════════════════════════════════════════════════════════
#  GRID DEFINITION
#  Each model runs independently. Total = N_GLOBAL x N_Mx per model.
#  All three are well under 20,000.
# ══════════════════════════════════════════════════════════════════════════════

GRID_GLOBAL = {
    "rv_window":            [120, 60],     # RV lookback bars
    "efficiency_window":    [20,  10],     # efficiency ratio window
    "efficiency_threshold": [0.3, 0.2],    # min directional efficiency
}
# N_GLOBAL = 8

GRID_M1 = {
    "asia_range_pct":    [25, 35, 45],    # asian range must be below this pct of history
    "body_mult":         [1.5, 1.0, 0.7], # body >= mult * median_body
    "tp_mult":           [3.0, 2.0, 1.5], # TP = entry +/- asia_range * mult
    "stop_buffer_atr":   [0.2, 0.0],      # extra ATR buffer on stop
}
# 8 x 54 = 432 combos

GRID_M2 = {
    "ldn_move_pct":      [80, 70, 60],    # london move must exceed this percentile
    "wick_mult":         [2.5, 2.0, 1.5], # wick >= mult * body
    "tp_multiple":       [3.0, 2.0, 1.5], # TP = entry +/- tp_mult * stop_dist
    "stop_buffer_atr":   [0.2, 0.0],
}
# 8 x 54 = 432 combos

GRID_M3 = {
    "box_lookback":      [60, 40, 20],    # consolidation window bars
    "box_atr_ratio":     [1.5, 1.0, 0.7], # box_range <= ATR * ratio
    "expansion_mult":    [1.5, 1.2, 1.0], # bar_range >= mult * median_range
    "tp_mult":           [3.0, 2.0, 1.5], # TP = entry +/- box_range * mult
}
# 8 x 81 = 648 combos

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
        p   = done / self.total if self.total else 1
        f   = int(self.w * p)
        el  = time.time() - self.t0
        eta = (el / p - el) if p > 0.001 else 0
        sys.stdout.write(
            f"\r{self.prefix} [{'█'*f}{'░'*(self.w-f)}] "
            f"{done}/{self.total} {p*100:.0f}%  {el:.0f}s  ETA {eta:.0f}s   "
        )
        sys.stdout.flush()
        if done >= self.total:
            sys.stdout.write("\n"); sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch(sym, n=FETCH_BARS):
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, n)
    if rates is None or len(rates) < 2000:
        logger.warning(f"  {sym}: fetch failed — {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    # ── IC Markets broker time → UTC ──────────────────────────────────────
    # IC Markets uses EET/EEST (Europe/Athens): UTC+2 winter, UTC+3 summer.
    # MT5 copy_rates returns Unix timestamps in BROKER LOCAL time (not UTC).
    # We must localize as Europe/Athens then convert to UTC so session
    # window logic (Asia 00:00, London 07:00, NY 12:30) is correct year-round.
    #
    # Note: MT5 timestamps are stored as seconds since epoch but represent
    # broker wall-clock time. So we first interpret them as broker-local,
    # then shift to UTC using the correct DST-aware offset.
    raw_ts = pd.to_datetime(df["time"], unit="s")   # naive, in broker local time
    try:
        # tz_localize interprets the naive timestamps as Europe/Athens local time
        # and handles DST transitions (ambiguous=infer resolves overlap at clock-back)
        broker_aware = raw_ts.dt.tz_localize(
            "Europe/Athens", ambiguous="infer", nonexistent="shift_forward"
        )
        utc_ts = broker_aware.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # Fallback if pytz/zoneinfo not available: use fixed UTC+2
        # This is wrong during EEST (summer) by 1 hour but avoids crash
        logger.warning("  pytz/zoneinfo DST conversion failed — falling back to UTC+2 fixed offset")
        utc_ts = raw_ts - pd.Timedelta(hours=2)
    df["time"]       = utc_ts
    df["h_utc"]      = df["time"].dt.hour
    df["m_utc"]      = df["time"].dt.minute
    df["date"]       = df["time"].dt.date
    df["min_of_day"] = df["h_utc"] * 60 + df["m_utc"]

    if "spread" in df.columns:
        info  = mt5.symbol_info(sym)
        point = info.point if info else 0.0001
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
#  BASE CACHE — computed once per symbol
#  Contains everything that does NOT depend on grid params.
#  rv_window and efficiency_window vary per combo → recomputed inside grid loop.
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
    tr  = np.maximum(h - l,
          np.maximum(np.abs(h - np.roll(c, 1)),
                     np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    # Body / range
    body      = np.abs(c - o)
    bar_range = h - l
    med_body  = pd.Series(body).rolling(100, min_periods=10).median().values
    med_range = pd.Series(bar_range).rolling(100, min_periods=10).median().values

    # Session masks — UTC minute-of-day, NOT bar counts
    asia_mask   = (mod >= 0)    & (mod < 360)    # 00:00-06:00
    london_mask = (mod >= 420)  & (mod < 600)    # 07:00-10:00 (M1 trade window)
    ny_mask     = (mod >= 750)  & (mod < 900)    # 12:30-15:00
    rollover    = (mod >= 1380) | (mod < 60)     # 23:00-01:00

    # ── Asian range per day (no lookahead: use today's range only within london window)
    unique_d    = np.unique(dt)
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
                if len(sorted_ar) > 0:
                    a_pct[i] = bisect.bisect_left(sorted_ar, r) / len(sorted_ar) * 100
                if d not in seen_ad:
                    bisect.insort(sorted_ar, r)
                    seen_ad.add(d)

    # ── London move per day: close of 11:59 bar minus open of 07:00 bar
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
                if len(sorted_lm) > 0:
                    ldn_mpct[i] = bisect.bisect_left(sorted_lm, ab) / len(sorted_lm) * 100
                if d not in seen_ld:
                    bisect.insort(sorted_lm, ab)
                    seen_ld.add(d)

    # log returns (needed for RV — stored raw, RV computed per rv_window in grid)
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
#  PARAM-DEPENDENT RECOMPUTE  (called once per global param combo)
# ══════════════════════════════════════════════════════════════════════════════

def compute_regimes_and_eff(base, rv_w, eff_w):
    n      = base["n"]
    c      = base["c"]
    lr     = base["log_ret"]

    # Realized volatility: sqrt(sum of squared log returns over rv_w bars)
    rv_raw = np.sqrt(np.maximum(
        pd.Series(lr ** 2).rolling(rv_w, min_periods=rv_w // 2).sum().values, 0.0))

    # Rolling historical percentile — expanding, no lookahead
    # Insert every 5th bar to keep the sorted list O(log n) manageable
    rv_pct    = np.full(n, 50.0)
    sorted_rv = []
    warmup    = max(rv_w * 3, 500)
    for i in range(n):
        if i >= warmup and len(sorted_rv) > 0:
            rv_pct[i] = bisect.bisect_left(sorted_rv, rv_raw[i]) / len(sorted_rv) * 100
        if i % 5 == 0:
            bisect.insort(sorted_rv, rv_raw[i])

    rl  = rv_pct < RV_LOW_PCT
    rm  = (rv_pct >= RV_MED_LO) & (rv_pct <= RV_MED_HI)
    rh  = rv_pct > RV_HIGH_PCT

    # Directional efficiency ratio
    shifted = pd.Series(c).shift(eff_w).values
    net     = np.abs(c - shifted)
    gross   = pd.Series(
        np.abs(np.diff(c, prepend=c[0]))
    ).rolling(eff_w, min_periods=2).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(gross > 0, net / gross, 0.0)

    return rl, rm, rh, eff


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTORS — vectorized, signal on bar i, entry on bar i+1
# ══════════════════════════════════════════════════════════════════════════════

def detect_m1(base, regime_med, eff, eff_thr, pm):
    """London Liquidity Expansion"""
    c   = base["c"]; o = base["o"]
    atr = base["atr"]
    body = base["body"]; med_body = base["med_body"]
    a_hi = base["a_hi"]; a_lo = base["a_lo"]
    a_rng = base["a_rng"]; a_pct = base["a_pct"]
    win  = base["london_mask"]

    valid = (win
             & regime_med
             & (eff >= eff_thr)
             & ~np.isnan(a_hi)
             & ~np.isnan(a_pct)
             & (a_pct < pm["asia_range_pct"])       # compressed session
             & (a_rng > 0)
             & (body >= pm["body_mult"] * med_body)) # displacement candle

    long_s  = valid & (c > a_hi)
    short_s = valid & (c < a_lo)

    n    = base["n"]
    dirs = np.zeros(n, dtype=np.int8)
    dirs[long_s]  = 1
    dirs[short_s] = -1

    buf  = pm["stop_buffer_atr"]
    sl   = np.full(n, np.nan)
    sl[long_s]  = a_lo[long_s]  - buf * atr[long_s]
    sl[short_s] = a_hi[short_s] + buf * atr[short_s]

    # TP distance array (per bar)
    tp_dist = a_rng * pm["tp_mult"]   # element-wise; index by sig_idx in backtest

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

    bear_rej = (bull_london
                & (c < o)
                & (upper_wick >= pm["wick_mult"] * safe_body))
    bull_rej = (bear_london
                & (c > o)
                & (lower_wick >= pm["wick_mult"] * safe_body))

    valid_s = win & regime_high & (eff >= eff_thr) & bear_rej
    valid_l = win & regime_high & (eff >= eff_thr) & bull_rej

    n    = base["n"]
    dirs = np.zeros(n, dtype=np.int8)
    dirs[valid_l] = 1
    dirs[valid_s] = -1

    buf  = pm["stop_buffer_atr"]
    sl   = np.full(n, np.nan)
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

    lkbk = pm["box_lookback"]
    # shift(1) on rolling window: box uses data up to bar i-1, no lookahead
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
    dirs[long_s]  = 1
    dirs[short_s] = -1

    box_mid = (box_h + box_l) / 2.0
    sl      = np.full(n, np.nan)
    sl[long_s]  = box_mid[long_s]
    sl[short_s] = box_mid[short_s]

    return dirs, sl, ("box_mult", box_r, pm["tp_mult"])


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZED BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def backtest_vec(base, dirs, sl_arr, tp_info, fixed_spread, model_id,
                 regime_low, regime_med, regime_high):
    n     = base["n"]
    o     = base["o"]; h = base["h"]; l = base["l"]; c = base["c"]
    sp    = base["sp"]
    times = base["times"]

    sig_idx = np.where(dirs != 0)[0]
    sig_idx = sig_idx[sig_idx + 1 < n]
    if len(sig_idx) == 0:
        return None, []

    ei  = sig_idx + 1                  # entry bar — no lookahead
    d   = dirs[sig_idx]

    # Spread-aware entry price
    sp_e = np.where(np.isnan(sp[ei]), fixed_spread, sp[ei])
    ep   = np.where(d == 1, o[ei] + sp_e / 2.0,
                             o[ei] - sp_e / 2.0)

    sl_p     = sl_arr[sig_idx]
    sl_dist  = np.maximum(np.abs(ep - sl_p), 1e-9)

    # Target price
    tp_type = tp_info[0]
    if tp_type == "fixed_dist":
        tp_dist_full = tp_info[1]   # full-length array indexed by sig_idx
        tp_d = tp_dist_full[sig_idx]
        tp_p = np.where(d == 1, ep + tp_d, ep - tp_d)
    elif tp_type == "sl_mult":
        mult = tp_info[1]
        tp_p = np.where(d == 1, ep + sl_dist * mult, ep - sl_dist * mult)
    else:  # "box_mult"
        box_r_full = tp_info[1]; mult = tp_info[2]
        br   = box_r_full[sig_idx]
        tp_p = np.where(d == 1, ep + br * mult, ep - br * mult)

    nt        = len(sig_idx)
    hold_caps = np.full(nt, MAX_HOLD_BARS, dtype=np.int32)   # uniform cap (no session boundary here)
    max_cap   = MAX_HOLD_BARS

    offsets = np.arange(max_cap)
    abs_i   = np.clip(ei[:, None] + offsets[None, :], 0, n - 1)  # (nt, max_cap)

    fut_h  = h[abs_i]; fut_l = l[abs_i]; fut_c = c[abs_i]
    sp_fwd = np.where(np.isnan(sp[abs_i]), fixed_spread, sp[abs_i])

    # off_mask: True only for valid offsets within each trade's hold cap
    off_mask = offsets[None, :] < hold_caps[:, None]             # (nt, max_cap)

    # SL hit: long uses low (bid), short uses high (ask) — intra-bar wick touches
    sl_hit = np.where(d[:, None] == 1,
                      fut_l <= sl_p[:, None],
                      fut_h + sp_fwd >= sl_p[:, None]) & off_mask
    # TP hit: long uses high (bid reaches target), short uses low (ask reaches target)
    tp_hit = np.where(d[:, None] == 1,
                      fut_h >= tp_p[:, None],
                      fut_l + sp_fwd <= tp_p[:, None]) & off_mask

    any_exit = sl_hit | tp_hit
    first    = np.argmax(any_exit, axis=1)
    hit      = any_exit[np.arange(nt), first]
    exit_off = np.where(hit, first, hold_caps - 1)

    at_sl  = sl_hit[np.arange(nt), exit_off]
    at_tp  = tp_hit[np.arange(nt), exit_off]
    exit_c = fut_c[np.arange(nt), exit_off]

    raw_r    = np.clip((exit_c - ep) / sl_dist * d, -2.0, 4.0)
    tp_rr    = np.abs(tp_p - ep) / sl_dist
    result_r = np.where(at_sl, -1.0, np.where(at_tp, tp_rr, raw_r))

    # Regime label per trade
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
    rm   = np.maximum.accumulate(bals)
    dd   = rm - bals
    mdd  = float(dd.max() / rm[np.argmax(dd)]) if len(bals) > 1 else 0.0

    t0  = pd.Timestamp(times[0]); t1 = pd.Timestamp(times[-1])
    wks = max((t1 - t0).total_seconds() / 604_800, 1.0)

    return {
        "total_trades":     tt,
        "win_rate":         round(wr,  4),
        "expectancy_r":     round(exp, 4),
        "profit_factor":    round(pf,  3),
        "max_drawdown":     round(mdd, 4),
        "trades_per_week":  round(tt / wks, 2),
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
    fixed_sp = 0.0002

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
                        fixed_sp, model_id, rl, rm, rh)
                except Exception:
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
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package not found.")
    if not mt5.initialize(login=LOGIN,
                          password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    logger.info("=" * 70)
    logger.info("MULTI-MODEL FX ENGINE — VECTORIZED GRID SEARCH")
    logger.info(f"M1 combos: {_ncombos(GRID_GLOBAL)*_ncombos(GRID_M1):,}  "
                f"M2: {_ncombos(GRID_GLOBAL)*_ncombos(GRID_M2):,}  "
                f"M3: {_ncombos(GRID_GLOBAL)*_ncombos(GRID_M3):,}  (all ≤20K)")
    logger.info("No lookahead: signal bar i → entry open[i+1]")
    logger.info("Spread-aware: long=open+sp/2  short=open-sp/2")
    logger.info("=" * 70)

    # Fetch
    base_caches = {}
    for sym in SYMBOLS:
        logger.info(f"\n[{sym}] fetching...")
        df = fetch(sym)
        if df is not None:
            logger.info(f"[{sym}] building cache...")
            base_caches[sym] = build_base_cache(df)

    mt5.shutdown()
    if not base_caches:
        logger.error("No data fetched. Aborting."); return

    # Models
    model_defs = [
        (1, detect_m1, GRID_M1, list(GRID_M1.keys())),
        (2, detect_m2, GRID_M2, list(GRID_M2.keys())),
        (3, detect_m3, GRID_M3, list(GRID_M3.keys())),
    ]

    for model_id, detect_fn, grid_m, keys_m in model_defs:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING MODEL {model_id} — {len(base_caches)} symbols")
        rows, trades = run_model_grid(model_id, detect_fn, grid_m, base_caches)

        if not rows:
            logger.warning(f"  Model {model_id}: zero valid combos — "
                           f"consider loosening thresholds")
            continue

        df_full = pd.DataFrame(rows)
        df_full.to_csv(f"mme_m{model_id}_grid.csv", index=False)

        if trades:
            pd.DataFrame(trades).to_csv(
                f"mme_m{model_id}_trades.csv", index=False)
            logger.info(f"  {len(trades):,} trades → mme_m{model_id}_trades.csv")

        agg_keys = list(GRID_GLOBAL.keys()) + keys_m
        agg = df_full.groupby(agg_keys).agg(
            symbols_valid   = ("symbol",          "count"),
            win_rate        = ("win_rate",         "mean"),
            expectancy_r    = ("expectancy_r",     "mean"),
            profit_factor   = ("profit_factor",    "mean"),
            max_drawdown    = ("max_drawdown",     "mean"),
            trades_per_week = ("trades_per_week",  "mean"),
        ).reset_index()

        agg = agg[agg["symbols_valid"] == len(base_caches)]
        agg = agg.sort_values("expectancy_r", ascending=False).reset_index(drop=True)
        agg.to_csv(f"mme_m{model_id}_agg.csv", index=False)
        logger.info(f"  Grid agg → mme_m{model_id}_agg.csv  "
                    f"({len(agg)} valid combos across all symbols)")

        print_top(agg, model_id, keys_m)

    logger.info(f"\n{'='*70}")
    logger.info("COMPLETE. Best params per model in mme_m1/m2/m3_agg.csv")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
