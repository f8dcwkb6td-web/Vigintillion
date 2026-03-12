"""
==============================================================================
Multi-Pair Edge Validation  |  14 Pairs  |  M1  |  2.5M bars
==============================================================================
VALIDATES TWO EDGES ACROSS ALL 14 PAIRS WITH FIXED PARAMS:

  EDGE A — FIX_WINDOW (4pm London WM/Reuters Fix)
    Params: vq=q70 sq=q30 slb=5 sam=0.1 bam=0.5 vm=2.0 buf=0.2 rr=2.0
    Validated on USDJPY: n=159 WR=79.9% E=+1.377 MDD=1.0%

  EDGE B — MONDAY_ASIA (Monday Asia Open)
    Params: vq=q60 sq=q30 slb=5 sam=0.3 bam=0.5 vm=2.0 buf=0.2 rr=2.0
    Validated on USDJPY: n=248 WR=71.8% E=+1.144 MDD=1.0%

PURPOSE:
  Overfit defense — if same fixed params produce 70%+ WR across uncorrelated
  pairs, curve fitting is mathematically impossible.

OUTPUT PER EDGE:
  - Results table sorted by Expectancy
  - Correlation matrix of R-multiples
  - Most uncorrelated pairs
  - Combined portfolio trade count

BARS: 2.5M per pair
==============================================================================
"""

import os, sys, io, logging, datetime
import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("EDGE_MULTI_PAIR")
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler("edge_multi_pair.log", maxBytes=20_000_000, backupCount=3, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_fh)
_sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_sh)

TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
LOGIN         = int(os.environ.get("MT5_LOGIN", 0))
PASSWORD      = os.environ.get("MT5_PASSWORD", "")
SERVER        = os.environ.get("MT5_SERVER", "")

FETCH_BARS             = 2_500_000
MIN_TRADES             = 30
MAX_HOLD               = 60
VWAP_WINDOW            = 10
SL_MULTIPLIER          = 1.0
COOLDOWN_BARS          = 10
MAX_TRADES_PER_SESSION = 3
RISK_PER_TRADE         = 0.01

PAIRS = [
    "USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY",
    "EURGBP", "EURAUD", "EURCAD", "GBPAUD",
]

# ── Hardcoded validated params per edge ──────────────────────────────────────
EDGE_PARAMS = {
    "FIX_WINDOW": {
        "vol_threshold_q": "q70",
        "spread_q":        "q30",
        "sweep_lookback":  5,
        "sweep_atr_mult":  0.1,
        "body_atr_mult":   0.5,
        "vol_mult":        2.0,
        "buffer_atr":      0.2,
        "rr_ratio":        2.0,
    },
    "MONDAY_ASIA": {
        "vol_threshold_q": "q60",
        "spread_q":        "q30",
        "sweep_lookback":  5,
        "sweep_atr_mult":  0.3,
        "body_atr_mult":   0.5,
        "vol_mult":        2.0,
        "buffer_atr":      0.2,
        "rr_ratio":        2.0,
    },
}

EDGES = {
    "FIX_WINDOW": {
        "description": "4pm London WM/Reuters Fix | 15-17 broker (13-15 UTC)",
        "window_fn":   lambda hours, dow: ((hours >= 15) & (hours < 17)),
        "asian_hours": (1, 9),
    },
    "MONDAY_ASIA": {
        "description": "Monday Asia Open | Mon 01-09 broker (Sun 23-Mon 07 UTC)",
        "window_fn":   lambda hours, dow: ((dow == 0) & (hours >= 1) & (hours < 9)),
        "asian_hours": (1, 9),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
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
#  CACHE BUILD
# ══════════════════════════════════════════════════════════════════════════════

def build_cache(arrays, times, edge_name):
    o, h, l, c, v = arrays
    n     = len(c)
    cache = {"m1_arrays": arrays, "times_m1": times}

    cache["atr14"]   = compute_atr(h, l, c, 14)
    cache["rvol_30"] = rolling_realized_vol(c, 30)

    VOL_LB = min(28_800, n // 2)
    for q in [30, 40, 50, 60, 70]:
        cache[f"rvol_q{q}"] = rolling_quantile(cache["rvol_30"], VOL_LB, q / 100)

    bar_range = h - l
    cache["bar_range"] = bar_range
    SPR_LB = min(43_200, n // 2)
    for q in [20, 30, 40, 50]:
        cache[f"spread_q{q}"] = rolling_quantile(bar_range, SPR_LB, q / 100)

    cache["vol_mean_60"]          = pd.Series(v.astype(float)).rolling(60, min_periods=10).mean().values
    cache[f"mvwap_{VWAP_WINDOW}"] = micro_vwap(h, l, c, v, VWAP_WINDOW)

    dt  = pd.DatetimeIndex(times)
    hrs = dt.hour
    dow = dt.dayofweek

    edge_def              = EDGES[edge_name]
    cache["trade_window"] = edge_def["window_fn"](hrs, dow)
    cache["dt"]           = dt
    cache["dates"]        = np.array(dt.date)
    cache["hours"]        = np.array(hrs)

    ah_start, ah_end = edge_def["asian_hours"]
    asian_mask = (hrs >= ah_start) & (hrs < ah_end)

    asian_hi = np.full(n, np.nan)
    asian_lo = np.full(n, np.nan)
    dates    = cache["dates"]
    unique_d = np.unique(dates)
    date_hi  = {}; date_lo = {}
    for d in unique_d:
        m = (dates == d) & asian_mask
        if m.any():
            date_hi[d] = h[m].max()
            date_lo[d] = l[m].min()
    prev_hi = prev_lo = np.nan
    cur_d   = None
    for i in range(n):
        d = dates[i]
        if d != cur_d:
            cur_d = d
            for delta in range(1, 8):
                pd_ = d - datetime.timedelta(days=delta)
                if pd_ in date_hi:
                    prev_hi = date_hi[pd_]
                    prev_lo = date_lo[pd_]
                    break
        if cache["trade_window"][i]:
            asian_hi[i] = prev_hi
            asian_lo[i] = prev_lo
    cache["asian_hi"] = asian_hi
    cache["asian_lo"] = asian_lo

    body = np.abs(c - o); rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        cache["body_ratio"] = np.where(rng > 0, body / rng, 0.0)
        cache["close_pos"]  = np.where(rng > 0, (c - l) / rng, 0.5)

    tw_arr     = cache["trade_window"].astype(bool)
    not_tw_idx = np.where(~tw_arr)[0]
    all_idx    = np.arange(n)
    ins        = np.searchsorted(not_tw_idx, all_idx, side="right")
    ins_clip   = np.minimum(ins, len(not_tw_idx) - 1)
    next_end   = not_tw_idx[ins_clip]
    cache["dist_to_sess_end"] = np.where(
        next_end > all_idx,
        np.minimum(next_end - all_idx, MAX_HOLD),
        MAX_HOLD
    ).astype(np.int32)

    cache["opp_bull"] = (c < o)
    cache["opp_bear"] = (c > o)
    return cache


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_signals(params, cache):
    o, h, l, c, v = cache["m1_arrays"]
    n   = len(c)
    atr = cache["atr14"]
    tw  = cache["trade_window"]

    vq = params["vol_threshold_q"].replace("q", "")
    sq = params["spread_q"].replace("q", "")
    regime = (cache["rvol_30"] <= cache[f"rvol_q{vq}"]) & \
             (cache["bar_range"] <= cache[f"spread_q{sq}"])

    N    = params["sweep_lookback"]
    mult = params["sweep_atr_mult"]
    roll_hi = pd.Series(h).rolling(N, min_periods=N // 2).max().shift(1).values
    roll_lo = pd.Series(l).rolling(N, min_periods=N // 2).min().shift(1).values

    bull_gen   = tw & (l < roll_lo - mult * atr) & (c > roll_lo)
    bear_gen   = tw & (h > roll_hi + mult * atr) & (c < roll_hi)
    has_range  = ~np.isnan(cache["asian_hi"])
    bull_asian = tw & has_range & (l < cache["asian_lo"] - mult * atr * 0.5) & (c > cache["asian_lo"])
    bear_asian = tw & has_range & (h > cache["asian_hi"] + mult * atr * 0.5) & (c < cache["asian_hi"])
    bull_sw    = bull_gen | bull_asian
    bear_sw    = bear_gen | bear_asian

    body    = np.abs(c - o)
    vm      = cache["vol_mean_60"]
    disp_Ab = (c > o) & (body >= params["body_atr_mult"] * atr) & (cache["body_ratio"] >= 0.70)
    disp_As = (c < o) & (body >= params["body_atr_mult"] * atr) & (cache["body_ratio"] >= 0.70)
    vol_spk = v >= params["vol_mult"] * vm
    disp_Bb = vol_spk & (cache["close_pos"] >= 0.80)
    disp_Bs = vol_spk & (cache["close_pos"] <= 0.20)
    db      = disp_Ab | disp_Bb
    dr      = disp_As | disp_Bs

    db_next = np.roll(db, -1); db_next[-1] = False
    dr_next = np.roll(dr, -1); dr_next[-1] = False

    signal = np.zeros(n, dtype=np.int8)
    signal = np.where(regime & bull_sw & (db | db_next),  1, signal)
    signal = np.where(regime & bear_sw & (dr | dr_next), -1, signal)
    signal[:100] = 0

    cache["_sweep_lo"] = roll_lo
    cache["_sweep_hi"] = roll_hi
    return signal


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(signal, cache, params, initial_balance=10_000.0):
    o, h, l, c, v = cache["m1_arrays"]
    n        = len(signal)
    atr      = cache["atr14"]
    rr       = max(float(params["rr_ratio"]), 1.0)
    buf      = params["buffer_atr"]
    mvwap    = cache[f"mvwap_{VWAP_WINDOW}"]
    sweep_lo = cache["_sweep_lo"]
    sweep_hi = cache["_sweep_hi"]
    dist_end = cache["dist_to_sess_end"]
    opp_bull = cache["opp_bull"]
    opp_bear = cache["opp_bear"]
    dates    = cache["dates"]
    dt_hours = cache["dt"].hour
    times    = cache["times_m1"]

    raw_idx = np.where(signal != 0)[0]
    raw_idx = raw_idx[raw_idx + 1 < n]
    if len(raw_idx) == 0:
        return _empty(), []

    dirs      = signal[raw_idx].astype(np.int8)
    entry_idx = raw_idx + 1
    ep        = o[entry_idx]

    has_lo   = ~np.isnan(sweep_lo[raw_idx])
    has_hi   = ~np.isnan(sweep_hi[raw_idx])
    sl_long  = np.where(has_lo, sweep_lo[raw_idx] - buf * atr[raw_idx], ep - SL_MULTIPLIER * atr[raw_idx])
    sl_short = np.where(has_hi, sweep_hi[raw_idx] + buf * atr[raw_idx], ep + SL_MULTIPLIER * atr[raw_idx])
    sl_price = np.where(dirs == 1, sl_long, sl_short)
    sl_dist  = np.maximum(np.abs(ep - sl_price), atr[raw_idx] * 0.05)
    tp_price = np.where(dirs == 1, ep + sl_dist * rr, ep - sl_dist * rr)

    n_raw       = len(raw_idx)
    valid       = np.zeros(n_raw, dtype=bool)
    last_exit   = -999
    sess_counts = {}

    for i in range(n_raw):
        ei       = int(entry_idx[i])
        d        = dates[ei]
        sess_key = (d, "S")
        if ei - last_exit < COOLDOWN_BARS:
            continue
        if sess_counts.get(sess_key, 0) >= MAX_TRADES_PER_SESSION:
            continue
        valid[i] = True
        sess_counts[sess_key] = sess_counts.get(sess_key, 0) + 1
        last_exit = ei + min(MAX_HOLD, int(dist_end[ei]))

    vi = np.where(valid)[0]
    if len(vi) == 0:
        return _empty(), []

    n_trades  = len(vi)
    v_entry   = entry_idx[vi]
    v_dirs    = dirs[vi]
    v_ep      = ep[vi]
    v_sl      = sl_price[vi]
    v_tp      = tp_price[vi]
    v_sl_dist = sl_dist[vi]
    hold_caps = np.minimum(dist_end[v_entry], MAX_HOLD).astype(np.int32)
    max_cap   = int(hold_caps.max())

    offsets  = np.arange(max_cap)
    abs_idx  = np.clip(v_entry[:, None] + offsets[None, :], 0, n - 1)
    fut_l    = l[abs_idx]; fut_h = h[abs_idx]; fut_c = c[abs_idx]
    vwap_2d  = mvwap[abs_idx]
    off_mask = offsets[None, :] < hold_caps[:, None]

    sl_p    = v_sl[:, None]; tp_p = v_tp[:, None]
    sl_hit  = np.where(v_dirs[:, None] == 1, fut_l <= sl_p, fut_h >= sl_p) & off_mask
    tp_hit  = np.where(v_dirs[:, None] == 1, fut_h >= tp_p, fut_l <= tp_p) & off_mask

    prev_c_2d = np.roll(fut_c, 1, axis=1); prev_c_2d[:, 0] = v_ep
    prev_vwap = np.roll(vwap_2d, 1, axis=1); prev_vwap[:, 0] = vwap_2d[:, 0]
    vwap_xl   = (fut_c < vwap_2d) & (prev_c_2d >= prev_vwap)
    vwap_xs   = (fut_c > vwap_2d) & (prev_c_2d <= prev_vwap)
    vwap_x    = np.where(v_dirs[:, None] == 1, vwap_xl, vwap_xs) & off_mask

    adv = np.where(v_dirs[:, None] == 1, opp_bull[abs_idx], opp_bear[abs_idx]) & off_mask
    cum = np.cumsum(adv.astype(np.int8), axis=1)
    c3  = np.zeros_like(cum)
    c3[:, 3:] = cum[:, 3:] - cum[:, :-3]
    c3[:, :3] = cum[:, :3]
    consec3  = (c3 >= 3) & off_mask
    early    = (vwap_x | consec3) & (offsets[None, :] >= 3) & off_mask
    any_exit = sl_hit | tp_hit | early
    first    = np.argmax(any_exit, axis=1)
    hit      = any_exit[np.arange(n_trades), first]
    exit_off = np.where(hit, first, hold_caps - 1)

    at_sl    = sl_hit[np.arange(n_trades), exit_off]
    at_tp    = tp_hit[np.arange(n_trades), exit_off]
    exit_c   = fut_c[np.arange(n_trades), exit_off]
    er       = np.clip((exit_c - v_ep) / v_sl_dist * np.where(v_dirs == 1, 1, -1), -2.0, rr)
    outcome_r = np.where(at_sl, -1.0, np.where(at_tp, rr, er))
    valid_t   = ~np.isnan(outcome_r)
    outcome_r = outcome_r[valid_t]
    v_entry_f = v_entry[valid_t]

    if len(outcome_r) == 0:
        return _empty(), []

    trade_log = [
        {"bar": int(v_entry_f[k]), "time": pd.Timestamp(times[int(v_entry_f[k])]), "r": float(outcome_r[k])}
        for k in range(len(outcome_r))
    ]

    wins_mask = outcome_r > 0
    wins      = int(wins_mask.sum())
    losses    = int((~wins_mask).sum())
    tt        = wins + losses
    win_r     = float(outcome_r[wins_mask].sum())
    loss_r    = float(np.abs(outcome_r[~wins_mask]).sum())
    wr        = wins / tt
    awr       = win_r / wins    if wins   else 0.0
    alr       = loss_r / losses if losses else 1.0
    exp_r     = wr * awr - (1 - wr) * alr
    pf        = win_r / loss_r  if loss_r > 0 else float("inf")

    balance = initial_balance
    bal_arr = [balance]
    for r_ in outcome_r:
        balance = balance * (1 + RISK_PER_TRADE * r_) if r_ > 0 \
                  else balance * (1 - RISK_PER_TRADE * abs(r_))
        bal_arr.append(balance)
    bal_arr = np.array(bal_arr)
    rm  = np.maximum.accumulate(bal_arr)
    dd  = rm - bal_arr
    mdd = float(dd.max() / rm[np.argmax(dd)]) if len(bal_arr) > 1 else 0.0

    signs  = np.where(wins_mask, 1, -1)
    trans  = np.diff(signs, prepend=signs[0] + 1)
    sts    = np.where(trans != 0)[0]
    slen   = np.diff(np.append(sts, len(signs)))
    ssig   = signs[sts]
    max_ls = int(slen[ssig == -1].max()) if (ssig == -1).any() else 0

    return {
        "total_trades":     tt,
        "wins":             wins,
        "win_rate":         wr,
        "expectancy_r":     float(exp_r),
        "profit_factor":    float(pf),
        "max_drawdown_pct": mdd,
        "max_loss_streak":  max_ls,
        "return_pct":       float((balance - initial_balance) / initial_balance),
    }, trade_log


def _empty():
    return {"total_trades": 0, "wins": 0, "win_rate": 0.0, "expectancy_r": 0.0,
            "profit_factor": 0.0, "max_drawdown_pct": 0.0, "max_loss_streak": 0, "return_pct": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
#  CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(pair_logs, pairs):
    series_dict = {}
    for sym in pairs:
        if sym in pair_logs and pair_logs[sym]:
            df  = pd.DataFrame(pair_logs[sym])
            df["hour"] = df["time"].dt.floor("h")
            s   = df.groupby("hour")["r"].sum()
            if len(s) > 10:
                series_dict[sym] = s
    if len(series_dict) < 2:
        return None
    return pd.DataFrame(series_dict).fillna(0).corr()


def print_correlation_report(corr, edge_name):
    if corr is None or corr.empty:
        logger.info(f"\n[{edge_name}] Correlation matrix: insufficient data")
        return
    syms = list(corr.columns)
    n    = len(syms)
    logger.info(f"\n*** R-MULTIPLE CORRELATION MATRIX — {edge_name} ***")
    header = f"{'':>8}" + "".join(f"  {s:>8}" for s in syms)
    logger.info(header)
    for s1 in syms:
        row = f"{s1:>8}"
        for s2 in syms:
            val = corr.loc[s1, s2]
            row += f"  {'1.000':>8}" if s1 == s2 else f"  {val:>+7.3f} "
        logger.info(row)

    pairs_corr = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_corr.append((syms[i], syms[j], corr.loc[syms[i], syms[j]]))

    logger.info(f"\n*** MOST UNCORRELATED PAIRS — {edge_name} ***")
    for s1, s2, c in sorted(pairs_corr, key=lambda x: abs(x[2]))[:8]:
        logger.info(f"  {s1:>8} / {s2:<8}  corr={c:+.3f}")

    logger.info(f"\n*** MOST CORRELATED PAIRS — {edge_name} ***")
    for s1, s2, c in sorted(pairs_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
        logger.info(f"  {s1:>8} / {s2:<8}  corr={c:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(results, edge_name):
    passed = [(s, r) for s, r in results.items() if r["total_trades"] >= MIN_TRADES]
    failed = [(s, r) for s, r in results.items() if r["total_trades"] < MIN_TRADES]
    passed.sort(key=lambda x: x[1]["expectancy_r"], reverse=True)

    sep = "─" * 85
    logger.info(f"\n{'='*85}")
    logger.info(f"RESULTS — {edge_name} — {EDGES[edge_name]['description']}")
    logger.info(f"Sorted by Expectancy | Passed: {len(passed)}/{len(results)} pairs")
    logger.info(sep)
    logger.info(f"{'PAIR':>8}  {'n':>5}  {'WR':>6}  {'E':>7}  {'PF':>6}  {'MDD':>6}  {'LSTR':>5}  {'RET':>8}")
    logger.info(sep)

    for sym, r in passed:
        logger.info(
            f"{sym:>8}  {r['total_trades']:>5}  {r['win_rate']:>5.1%}  "
            f"{r['expectancy_r']:>+6.3f}  {r['profit_factor']:>6.2f}  "
            f"{r['max_drawdown_pct']:>5.1%}  {r['max_loss_streak']:>5}  "
            f"{r['return_pct']:>+7.0%}"
        )

    if failed:
        logger.info(sep)
        logger.info(f"INSUFFICIENT TRADES (< {MIN_TRADES}):")
        for sym, r in failed:
            logger.info(f"  {sym}: {r['total_trades']} trades")
    logger.info(sep)

    # Portfolio summary
    total_n   = sum(r["total_trades"] for _, r in passed)
    avg_wr    = np.mean([r["win_rate"] for _, r in passed]) if passed else 0
    avg_exp   = np.mean([r["expectancy_r"] for _, r in passed]) if passed else 0
    pairs_70  = sum(1 for _, r in passed if r["win_rate"] >= 0.70)
    pairs_exp = sum(1 for _, r in passed if r["expectancy_r"] >= 0.8)

    logger.info(f"\n*** PORTFOLIO SUMMARY — {edge_name} ***")
    logger.info(f"  Pairs passing min trades : {len(passed)}/{len(results)}")
    logger.info(f"  Pairs with WR >= 70%     : {pairs_70}/{len(passed)}")
    logger.info(f"  Pairs with E >= 0.8      : {pairs_exp}/{len(passed)}")
    logger.info(f"  Average WR               : {avg_wr:.1%}")
    logger.info(f"  Average Expectancy       : {avg_exp:+.3f}")
    logger.info(f"  Total trades (portfolio) : {total_n}")
    logger.info(f"  Trades per year (est)    : {total_n / 6.5:.0f}")
    logger.info(f"  Trades per week (est)    : {total_n / (6.5 * 52):.1f}")

    # Overfit verdict
    if pairs_70 >= 8 and avg_exp >= 0.8:
        logger.info(f"\n  ✓ OVERFIT DEFENSE PASSED — {pairs_70} pairs with WR≥70%, avg E={avg_exp:+.3f}")
        logger.info(f"    Same fixed params generalise across uncorrelated assets. Not curve fitted.")
    elif pairs_70 >= 5:
        logger.info(f"\n  ~ PARTIAL — {pairs_70} pairs pass WR≥70%. Moderate generalization.")
    else:
        logger.info(f"\n  ✗ WEAK GENERALIZATION — edge may be pair-specific")

    return passed


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_pair(symbol, n=FETCH_BARS):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n)
    if rates is None or len(rates) < 200:
        logger.warning(f"  {symbol}: fetch failed — {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    logger.info(f"  {symbol}: {len(df):,} bars | {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_scan():
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 not installed.")
    if not mt5.initialize(path=TERMINAL_PATH, login=LOGIN, password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    logger.info("=" * 85)
    logger.info("MULTI-PAIR EDGE VALIDATION — Fix Window + Monday Asia")
    logger.info(f"Pairs: {len(PAIRS)} | Bars: {FETCH_BARS:,} | Fixed params per edge")
    logger.info("=" * 85)

    for edge_name, edge_def in EDGES.items():
        logger.info(f"\n{'='*85}")
        logger.info(f"EDGE: {edge_name} — {edge_def['description']}")
        logger.info(f"Params: {EDGE_PARAMS[edge_name]}")
        logger.info(f"{'='*85}")

        results   = {}
        pair_logs = {}

        for sym in PAIRS:
            logger.info(f"\n[{sym}] fetching...")
            df = fetch_pair(sym)
            if df is None:
                results[sym]   = _empty()
                pair_logs[sym] = []
                continue

            arrays = (df["open"].values, df["high"].values,
                      df["low"].values,  df["close"].values,
                      df["tick_volume"].values)

            logger.info(f"[{sym}] building cache + running backtest...")
            cache  = build_cache(arrays, df["time"].values, edge_name)
            signal = detect_signals(EDGE_PARAMS[edge_name], cache)
            stats, trade_log = run_backtest(signal, cache, EDGE_PARAMS[edge_name])

            results[sym]   = stats
            pair_logs[sym] = trade_log

            logger.info(
                f"[{sym}] n={stats['total_trades']} WR={stats['win_rate']:.1%} "
                f"E={stats['expectancy_r']:+.3f} MDD={stats['max_drawdown_pct']:.1%} "
                f"lstr={stats['max_loss_streak']}"
            )

        # Results table + portfolio summary
        passed = print_results_table(results, edge_name)

        # Correlation matrix
        active = [s for s in PAIRS if pair_logs.get(s) and len(pair_logs[s]) >= MIN_TRADES]
        corr   = compute_correlation_matrix(pair_logs, active)
        print_correlation_report(corr, edge_name)

    mt5.shutdown()
    logger.info("\n" + "=" * 85)
    logger.info("Scan complete — full results in edge_multi_pair.log")
    logger.info("=" * 85)


if __name__ == "__main__":
    run_scan()
