import ctypes
# Prevent Windows from sleeping
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
# At top of file
trade_stats = {
    "total_trades": 0,
    "wins": 0
}
import logging

# --- Fix duplicate logging ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from datetime import datetime

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler("undecillion.log"),  # Log to file
        logging.StreamHandler()                 # Also log to console
    ]
)


import MetaTrader5 as mt5

# Replace with your credentials
login_id =7376407
password = "iC8XoiRp&4L4KU"
server = "ICMarketsSC-MT5-2"

mt5.shutdown()  # Clean slate
if not mt5.initialize():
    raise Exception("MT5 initialize failed")

if not mt5.login(login_id, password, server):
    raise Exception("MT5 login failed")

account_info = mt5.account_info()
if account_info is None:
    raise Exception("Could not retrieve account info after login")

print(f" Logged into account {account_info.login} | Balance: {account_info.balance} | Margin Free: {account_info.margin_free} | Margin Level: {account_info.margin_level}")

# === CONFIGURATION ===
symbols = ["USDJPY"]


lot_step = 0.01

slippage = 3
magic_number = 20250708  # Unique identifier for the Undecillion engine
timezone_offset = -5  # Jamaica timezone is UTC-5

session_times = {

    "london": {"start": 3, "end": 10},    # 03:00 to 12:00
    "ny": {"start": 8, "end": 16},        # 08:00 to 17:00
}




import MetaTrader5 as mt5

# === INIT MT5 with explicit VPS terminal path ===
terminal_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # <-- replace this with your VPS MT5 path

if not mt5.initialize(path=terminal_path):
    raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

def safe_str(value, fmt="{:.5f}"):
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value) if value is not None else "N/A"



import logging


def is_time_in_session(hour, start, end):
    """Check if hour is within a session that may cross midnight."""
    if start <= end:
        return start <= hour < end
    else:
        # Session crosses midnight
        return hour >= start or hour < end

def is_valid_session():
    now = datetime.now()
    hour = now.hour

    sessions = {

        "london": [(24, 10)],           # 3AM–12PM
        "ny": [(8, 24 )],               # 8AM–5PM
    }

    return any(
        is_time_in_session(hour, start, end)
        for session_periods in sessions.values()
        for start, end in session_periods
    )
def place_trade(symbol, direction, entry, sl, tp, volume, slippage, magic_number):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"[{symbol}] Symbol info not found.")
        return None

    point = symbol_info.point
    digits = symbol_info.digits
    min_stop_dist = symbol_info.trade_stops_level * point

    if not mt5.symbol_select(symbol, True):
        logging.error(f"[{symbol}] Failed to select symbol in market watch.")
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"[{symbol}] Failed to get tick data.")
        return None

    market_price = tick.ask if direction == "buy" else tick.bid
    entry = round(entry, digits)
    sl = round(sl, digits)
    tp = round(tp, digits)
    market_price = round(market_price, digits)

    # === SL/TP Validation ===
    if direction == "buy":
        if not (sl < entry < tp):
            logging.error(f"[{symbol}] Invalid SL/TP for BUY: SL={sl}, Entry={entry}, TP={tp}")
            return None
        if (entry - sl) < min_stop_dist:
            sl = round(entry - min_stop_dist, digits)
            logging.warning(f"[{symbol}] Adjusted SL to meet minimum stop distance: {sl}")
        if (tp - entry) < min_stop_dist:
            tp = round(entry + min_stop_dist, digits)
            logging.warning(f"[{symbol}] Adjusted TP to meet minimum stop distance: {tp}")
    else:
        if not (tp < entry < sl):
            logging.error(f"[{symbol}] Invalid SL/TP for SELL: TP={tp}, Entry={entry}, SL={sl}")
            return None
        if (sl - entry) < min_stop_dist:
            sl = round(entry + min_stop_dist, digits)
            logging.warning(f"[{symbol}] Adjusted SL to meet minimum stop distance: {sl}")
        if (entry - tp) < min_stop_dist:
            tp = round(entry - min_stop_dist, digits)
            logging.warning(f"[{symbol}] Adjusted TP to meet minimum stop distance: {tp}")

    # === Volume Sanity Check ===
    if volume < symbol_info.volume_min:
        logging.warning(f"[{symbol}] Volume {volume} below minimum. Adjusted to {symbol_info.volume_min}")
        volume = symbol_info.volume_min
    elif volume > symbol_info.volume_max:
        logging.warning(f"[{symbol}] Volume {volume} above maximum. Adjusted to {symbol_info.volume_max}")
        volume = symbol_info.volume_max

    volume = round(volume, 2)

    # === Trade Request ===
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
        "price": entry,  # Use model-defined entry, not market price
        "sl": sl,
        "tp": tp,
        "deviation": slippage,
        "magic": magic_number,
        "comment": f"Undecillion auto trade {symbol}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logging.info(f"[{symbol}] Sending {direction.upper()} order: Entry={entry}, SL={sl}, TP={tp}, Vol={volume}")
    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(f"[{symbol}] Trade executed: {direction.upper()} {volume} lots at {entry}")
        acc = mt5.account_info()
        if acc:
            logging.info(f"[{symbol}] Balance: {acc.balance}, Free Margin: {acc.margin_free}, Margin Level: {acc.margin_level}")
        return pd.DataFrame([{
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "sl": sl,
            "tp": tp,
            "volume": volume,
            "magic": magic_number,
            "timestamp": datetime.now(),
        }])
    else:
        logging.error(f"[{symbol}] Trade failed | retcode={result.retcode} | comment={result.comment} | request={request}")
        return {
            "symbol": symbol,
            "error": "trade_failed",
            "retcode": result.retcode,
            "comment": result.comment,
            "request": request,
        }

# Example usage of session check
if __name__ == "__main__":
    if is_valid_session():
        print("We are inside a trading session.")
    else:
        print("Out of trading sessions now.")


    def generate_core_signals(df):
        import pandas as pd
        import numpy as np

        # Work on a copy to avoid side-effects
        df = df.copy()

        # Initialize output columns (preserve original names & defaults)
        for col in ["signal_flag", "direction", "entry_price", "sl", "tp", "signal_reason",
                    "pattern_match_score", "pattern_build_score", "pre_signal_bias", "pattern_id"]:
            if col not in df.columns:
                if col in ("signal_flag", "pattern_match_score", "pattern_build_score"):
                    df[col] = 0
                else:
                    df[col] = None

        # Ensure numeric + fill gaps
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.ffill().bfill()

        # Candle attributes (vectorized)
        df['body_size'] = (df['close'] - df['open']).abs()
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['bull_bear_ratio'] = (df['close'] - df['open']) / (df['candle_range'] + 1e-9)

        # Precompute rolling volume ratio (base value; may be overwritten for signaled rows)
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(10).mean() + 1e-9)

        # Precompute ATR14 if missing — use .bfill() to avoid deprecated fillna(method=...)
        if 'atr14' not in df.columns:
            df['atr14'] = df['candle_range'].rolling(14, min_periods=8).mean().bfill()

        eps = 1e-9
        n = len(df)

        # Next candle values (used for entry_price and confirmation)
        next_open = df['open'].shift(-1)
        next_close = df['close'].shift(-1)
        next_open_valid = next_open.notna()

        next_body_series = (next_close - next_open).abs()
        next_dir_bull_series = next_close > next_open
        next_dir_bear_series = next_close < next_open

        # ATR as float series
        atr_series = df['atr14'].astype(float)

        # --- Prepare result Series with stable dtypes (prevents dtype warnings) ---
        signal_flag = pd.Series(0, index=df.index, dtype='int8')
        direction = pd.Series([None] * n, index=df.index, dtype=object)
        entry_price = pd.Series(np.nan, index=df.index, dtype=float)
        sl = pd.Series(np.nan, index=df.index, dtype=float)
        tp = pd.Series(np.nan, index=df.index, dtype=float)
        signal_reason = pd.Series([None] * n, index=df.index, dtype=object)
        pattern_build_score = pd.Series(0.0, index=df.index, dtype=float)
        pre_signal_bias = pd.Series([None] * n, index=df.index, dtype=object)
        pattern_id = pd.Series([None] * n, index=df.index, dtype=object)


        # -----------------------
        # MSB Sell (Market Structure Break) - Tunable SL/TP (vectorized)
        # -----------------------
        LOOKBACK = 8
        CONFIRM_BODY_ATR = 0.3
        SL_MULT = 1.7
        TP_MULT = 1.7

        recent_low = df['low'].rolling(window=LOOKBACK, min_periods=LOOKBACK).min().shift(1)
        cond_msb_base = (df['close'] < recent_low) & (df['close'] < df['open'])
        cond_msb_confirm = next_dir_bear_series & (next_body_series >= (CONFIRM_BODY_ATR * atr_series))

        msb_mask = (cond_msb_base & cond_msb_confirm & next_open_valid).fillna(False)

        if msb_mask.any():
            idxs = msb_mask[msb_mask].index
            entry_vals = next_open.loc[idxs].astype(float)
            sig_low_vals = df['low'].loc[idxs].astype(float)
            sig_high_vals = df['high'].loc[idxs].astype(float)

            sl_vals = (entry_vals + (SL_MULT * atr_series.loc[idxs])).round(3)
            tp_vals = (entry_vals - (TP_MULT * atr_series.loc[idxs])).round(3)

            signal_flag.loc[idxs] = 1
            direction.loc[idxs] = "sell"
            entry_price.loc[idxs] = entry_vals
            sl.loc[idxs] = sl_vals
            tp.loc[idxs] = tp_vals
            signal_reason.loc[idxs] = "MSB Sell"
            pattern_build_score.loc[idxs] = 1.0
            pre_signal_bias.loc[idxs] = "sell"
            pattern_id.loc[idxs] = "MSB_Sell"
        # -----------------------
        # LSR Buy (Liquidity Sweep Reclaim) - High-Confidence Buy
        # -----------------------
        LOOKBACK = 15
        CONFIRM_BODY_ATR = 0.6
        SL_MULT = 1.9
        TP_MULT = 2.5

        recent_low = df['low'].rolling(window=LOOKBACK, min_periods=LOOKBACK).min().shift(1)

        # Sweep below prior liquidity
        cond_sweep = df['low'] < recent_low

        # Reclaim: bullish close back above sweep zone
        cond_reclaim = (df['close'] > recent_low) & (df['close'] > df['open'])

        # Confirmation candle
        cond_confirm = next_dir_bull_series & (next_body_series >= (CONFIRM_BODY_ATR * atr_series))

        lsr_buy_mask = (cond_sweep & cond_reclaim & cond_confirm & next_open_valid).fillna(False)

        if lsr_buy_mask.any():
            idxs = lsr_buy_mask[lsr_buy_mask].index
            entry_vals = next_open.loc[idxs].astype(float)

            sl_vals = (entry_vals - (SL_MULT * atr_series.loc[idxs])).round(3)
            tp_vals = (entry_vals + (TP_MULT * atr_series.loc[idxs])).round(3)

            signal_flag.loc[idxs] = 1
            direction.loc[idxs] = "buy"
            entry_price.loc[idxs] = entry_vals
            sl.loc[idxs] = sl_vals
            tp.loc[idxs] = tp_vals
            signal_reason.loc[idxs] = "LSR Buy"
            pattern_build_score.loc[idxs] = 1.15
            pre_signal_bias.loc[idxs] = "buy"
            pattern_id.loc[idxs] = "LSR_Buy"


        # -----------------------
        # Commit results into dataframe (vectorized assignment)
        # -----------------------
        df['signal_flag'] = signal_flag
        df['direction'] = direction
        df['entry_price'] = entry_price
        df['sl'] = sl
        df['tp'] = tp
        df['signal_reason'] = signal_reason
        df['pattern_build_score'] = pattern_build_score
        df['pre_signal_bias'] = pre_signal_bias
        df['pattern_id'] = pattern_id

        # Ensure volume_ratio uses lookback=4 like original for signaled rows
        vol_mean_lookback4 = df['volume'].rolling(window=4, min_periods=1).mean()
        signaled_idx = df.index[df['signal_flag'] == 1]
        if len(signaled_idx) > 0:
            df.loc[signaled_idx, 'volume_ratio'] = (df.loc[signaled_idx, 'volume'] /
                                                    (vol_mean_lookback4.loc[signaled_idx] + eps))

        # Final column to mirror original end-of-function step
        df['entry_signal'] = df['signal_flag']

        return df




def apply_trap_mapping(df):
    """
    Enhanced Module 2: Multi-Layer Trap Mapping v2 (Fixed Logic).
    """

    df = df.copy()
    df["trap_map_score"] = 0.0
    df["trap_reject"] = False

    for i in range(10, len(df)):
        idx = df.index[i]  # get index label

        if df.at[idx, "signal_flag"] != 1:
            continue  # Only analyze potential signals

        # Candle Info
        o, h, l, c, v = df.iloc[i][["open", "high", "low", "close", "volume"]]
        prev = df.iloc[i - 1]
        body = abs(c - o)
        candle_range = h - l
        wick_top = h - max(c, o)
        wick_bot = min(c, o) - l
        direction = df.at[idx, "direction"]
        session = df.at[idx, "session"] if "session" in df.columns else "unknown"
        trap_score = 0

        # Trap Patterns
        if direction == "buy" and wick_bot > 0.5 * candle_range:
            trap_score += 1
        elif direction == "sell" and wick_top > 0.5 * candle_range:
            trap_score += 1

        if direction == "buy" and prev["close"] < prev["open"] and abs(prev["close"] - prev["open"]) > body:
            trap_score += 1
        elif direction == "sell" and prev["close"] > prev["open"] and abs(prev["close"] - prev["open"]) > body:
            trap_score += 1

        if candle_range > 0 and body / candle_range < 0.2 and wick_top > 0.4 * candle_range and wick_bot > 0.4 * candle_range:
            trap_score += 1

        volume_mean = df.iloc[i-5:i]["volume"].mean()
        if v > 1.8 * volume_mean:
            trap_score += 1

        recent_high = df.iloc[i-3:i]["high"].max()
        recent_low = df.iloc[i-3:i]["low"].min()
        if direction == "buy" and h > recent_high and c < recent_high:
            trap_score += 1
        elif direction == "sell" and l < recent_low and c > recent_low:
            trap_score += 1

        net_move = df["close"].iloc[i - 6] - df["open"].iloc[i - 6]
        if direction == "buy" and net_move > 0.7 and prev["close"] < prev["open"]:
            trap_score += 1
        elif direction == "sell" and net_move < -0.7 and prev["close"] > prev["open"]:
            trap_score += 1

        # Scoring (Corrected)
        df.at[idx, "trap_map_score"] = min(1.0, trap_score / 5.0)
        df.at[idx, "trap_reject"] = trap_score >= 4  # Only reject strong traps

    return df


def apply_liquidity_filter(df):
    """
    Module 3: Liquidity Exposure Zone Filtering (Fixed Version).
    """

    df = df.copy()
    df["liquidity_score"] = 0.0
    df["liquidity_reject"] = False

    lookback_range = 500
    zone_touch_threshold = 0.0015
    round_level_precision = 0.005
    wick_sensitivity = 0.65

    for i in range(lookback_range, len(df)):
        idx = df.index[i]  # use index label

        if df.at[idx, "signal_flag"] != 1:
            continue  # No fake liquidity scores

        direction = df.at[idx, "direction"]
        entry_price = df.at[idx, "entry_price"]
        recent = df.iloc[i - lookback_range : i]

        # Clusters
        high_clusters = recent["high"].round(5).value_counts()
        low_clusters = recent["low"].round(5).value_counts()
        liq_highs = high_clusters[high_clusters >= 3].index.tolist()
        liq_lows = low_clusters[low_clusters >= 3].index.tolist()

        # Round Levels
        round_levels = list(set([
            round(lvl, 3)
            for lvl in recent["high"].tolist() + recent["low"].tolist()
            if abs((lvl * 1000) % (round_level_precision * 1000)) < 0.01
        ]))

        # Wick Zones
        wick_zones = []
        for _, row in recent.iterrows():
            body = abs(row["open"] - row["close"])
            total = row["high"] - row["low"]
            if total == 0:
                continue
            wick_top = row["high"] - max(row["open"], row["close"])
            wick_bot = min(row["open"], row["close"]) - row["low"]
            if direction == "buy" and wick_top / total > wick_sensitivity:
                wick_zones.append(row["high"])
            elif direction == "sell" and wick_bot / total > wick_sensitivity:
                wick_zones.append(row["low"])

        # Proximity Scoring
        nearby_liq = []

        if direction == "buy":
            nearby_liq += [lvl for lvl in liq_highs if 0 < lvl - entry_price < zone_touch_threshold]
            nearby_liq += [lvl for lvl in round_levels if 0 < lvl - entry_price < zone_touch_threshold]
            nearby_liq += [lvl for lvl in wick_zones if 0 < lvl - entry_price < zone_touch_threshold]
        elif direction == "sell":
            nearby_liq += [lvl for lvl in liq_lows if 0 < entry_price - lvl < zone_touch_threshold]
            nearby_liq += [lvl for lvl in round_levels if 0 < entry_price - lvl < zone_touch_threshold]
            nearby_liq += [lvl for lvl in wick_zones if 0 < entry_price - lvl < zone_touch_threshold]

        proximity_score = sum([1 - abs(entry_price - z) / zone_touch_threshold for z in nearby_liq])
        df.at[idx, "liquidity_score"] = min(1.0, proximity_score)
        df.at[idx, "liquidity_reject"] = proximity_score >= 0.7
    # === Long-Term Ceiling Protection ===
    long_lookback = 1000  # bars to consider for multi-month high
    ceiling_threshold = 0.0005  # price distance threshold (adjust to your symbol precision)

    for i in range(lookback_range, len(df)):
        idx = df.index[i]

        direction = df.at[idx, "direction"]
        entry_price = df.at[idx, "entry_price"]

        # Defensive: skip if data missing or NaN
        if pd.isna(entry_price) or direction not in ["buy", "sell"]:
            continue

        # Calculate long-term high for prior bars safely
        start_idx = max(0, i - long_lookback)
        long_term_high = df["high"].iloc[start_idx:i].max()

        if direction == "buy" and abs(entry_price - long_term_high) < ceiling_threshold:
            df.at[idx, "liquidity_reject"] = True
            df.at[idx, "liquidity_score"] = 1.0  # max penalty to strongly reject

    return df





def undecillion_signal_pipeline(df):
    """
    Undecillion signal pipeline (Modules 1–3 only):
    - Applies: Core Signals, Trap Mapping, Liquidity Filtering
    - Minimal penalties from trap/liquidity rejections
    - Signal rejected only if final_score < 0.7
    """

    df = generate_core_signals(df)
    df = apply_trap_mapping(df)
    df = apply_liquidity_filter(df)

    if "pattern_match_score" not in df.columns:
        df["pattern_match_score"] = 0.0
    if "signal_reason" not in df.columns:
        df["signal_reason"] = ""

    # === Soft penalties with minimal impact ===
    df["final_score"] = df["pattern_match_score"]

    df.loc[df["trap_reject"], "final_score"] *= 0.95    # Was 0.85
    df.loc[df["liquidity_reject"], "final_score"] *= 0.90  # Was 0.70

    # === Final rejection logic ===
    df["final_reject"] = df["final_score"] < 0.7

    mask = (df["signal_flag"] == 1) & (df["final_reject"])
    df.loc[mask, ["signal_flag", "direction", "entry_price", "sl", "tp"]] = [0, None, None, None, None]
    df.loc[mask, "signal_reason"] += " | Rejected due to low confidence score."

    return df

def apply_mvrf_filter(df):
    """
    Module 4: Micro-Volatility Reflex Filters (MVRF)
    Detects micro-wickout zones, fake breakouts, volatility anomalies.
    """

    df = df.copy()
    df["mvrf_score"] = 0
    df["mvrf_reject"] = False

    lookback = 6
    body_threshold = 0.25
    wick_threshold = 0.55
    volatility_spike_multiplier = 1.6
    max_allowed_anomalies = 3

    for i in range(lookback, len(df)):
        idx = df.index[i]  # get index label for .at usage

        if df.at[idx, "signal_flag"] != 1:
            continue

        direction = df.at[idx, "direction"]
        if direction not in ["buy", "sell"]:
            continue

        recent = df.iloc[i - lookback:i]
        mvrf_score = 0
        ranges = []

        for j in range(lookback):
            row = recent.iloc[j]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            body = abs(c - o)
            range_ = h - l
            if range_ == 0:
                continue
            ranges.append(range_)

            body_ratio = body / range_
            wick_top = h - max(c, o)
            wick_bottom = min(c, o) - l
            wick_top_ratio = wick_top / range_
            wick_bottom_ratio = wick_bottom / range_

            if direction == "buy" and wick_top_ratio > wick_threshold and body_ratio < body_threshold:
                mvrf_score += 1
            elif direction == "sell" and wick_bottom_ratio > wick_threshold and body_ratio < body_threshold:
                mvrf_score += 1

            if wick_top_ratio > 0.4 and wick_bottom_ratio > 0.4 and body_ratio < 0.2:
                mvrf_score += 1

            if direction == "buy" and wick_bottom_ratio < 0.1 and wick_top_ratio > 0.5:
                mvrf_score += 1
            elif direction == "sell" and wick_top_ratio < 0.1 and wick_bottom_ratio > 0.5:
                mvrf_score += 1

            if j > 0:
                prev = recent.iloc[j - 1]
                prev_body = prev["close"] - prev["open"]
                curr_body = c - o
                if (prev_body > 0 > curr_body or prev_body < 0 < curr_body) and abs(curr_body) < abs(prev_body) * 0.6:
                    if h - l < 0.0025:
                        mvrf_score += 1

        # Volatility spike detection
        if len(ranges) >= 2:
            last_range = ranges[-1]
            avg_prev = np.mean(ranges[:-1])
            if avg_prev > 0 and last_range > avg_prev * volatility_spike_multiplier:
                last_row = recent.iloc[-1]
                last_body = abs(last_row["open"] - last_row["close"])
                if last_body < 0.25 * last_range:
                    mvrf_score += 1

        # Directional wick hits
        wick_hits = 0
        for row in recent.itertuples():
            full_range = row.high - row.low
            if full_range == 0:
                continue
            wick_top = row.high - max(row.open, row.close)
            wick_bottom = min(row.open, row.close) - row.low
            if direction == "buy" and wick_top / full_range > wick_threshold:
                wick_hits += 1
            elif direction == "sell" and wick_bottom / full_range > wick_threshold:
                wick_hits += 1
        if wick_hits >= 2:
            mvrf_score += 1

        # Compressed volatility detection
        range_std = np.std(ranges) if ranges else 0
        avg_price = np.mean([row.close for row in recent.itertuples()]) if len(recent) > 0 else 0
        std_normalized = range_std / avg_price if avg_price > 0 else 0
        if std_normalized < 0.00025:
            mvrf_score += 1

        # Cap score
        mvrf_score = min(mvrf_score, 6)
        df.at[idx, "mvrf_score"] = mvrf_score
        df.at[idx, "mvrf_reject"] = mvrf_score > max_allowed_anomalies

    return df




def undecillion_signal_pipeline_with_mvrf(df):
    """
    Full Undecillion pipeline including Module 4 (MVRF) and Module 8 (Wick SL Protection)
    with signal reason propagation and rejection handling.
    """

    df = generate_core_signals(df)
    df = apply_trap_mapping(df)
    df = apply_liquidity_filter(df)
    df = apply_mvrf_filter(df)

    # Combine final rejection from all filters
    df["final_reject"] = df["trap_reject"] | df["liquidity_reject"] | df["mvrf_reject"]

    # Mark wick protection column default
    df["wick_protected"] = False

    # Remove or nullify rejected signals
    df.loc[df["final_reject"], "signal_flag"] = 0
    df.loc[df["final_reject"], "direction"] = None
    df.loc[df["final_reject"], "entry_price"] = None
    df.loc[df["final_reject"], "sl"] = None
    df.loc[df["final_reject"], "tp"] = None
    df.loc[df["final_reject"], "signal_reason"] += " | Rejected by trap/liquidity/mvrf filters."
    return df
    # Apply more aggressive wick protection on passed signals




def apply_reclaim_layer(df):
    """
    Module 5: Reclaim Layer — Re-enable signals previously rejected but
    which show strong agreement across patterns and liquidity/trap filters
    exactly as per screenshot session logic.
    """

    # Add reclaim columns if missing
    if "reclaim_flag" not in df.columns:
        df["reclaim_flag"] = False
    if "reclaim_reason" not in df.columns:
        df["reclaim_reason"] = ""

    for i in range(len(df)):
        # Only consider candles without signals
        if df.at[i, "signal_flag"] == 0:
            # Check if the pattern (Module 1), trap (Module 2), liquidity (Module 3), and MVRF (Module 4) agree
            pattern_agree = df.at[i, "pattern_match_score"] if "pattern_match_score" in df.columns else 0
            trap_agree = df.at[i, "trap_map_score"] if "trap_map_score" in df.columns else 0
            liquidity_agree = df.at[i, "liquidity_agree"] if "liquidity_agree" in df.columns else 0
            mvrf_agree = 1 if ("mvrf_reject" in df.columns and df.at[i, "mvrf_reject"] == False) else 0

            # Strict reclaim condition: all agree fully (as in screenshots)
            if pattern_agree >= 1.0 and trap_agree >= 1.0 and liquidity_agree == 1 and mvrf_agree == 1:
                df.at[i, "signal_flag"] = 1
                df.at[i, "reclaim_flag"] = True
                df.at[i, "reclaim_reason"] = "Reclaimed by strong model agreement (pattern, trap, liquidity, MVRF)."

    reclaim_count = df["reclaim_flag"].sum()
    if reclaim_count > 0:
        logging.info(f"Module 5: Reclaimed {reclaim_count} previously rejected signals.")

    return df
# === META CORTEX: Signal Confidence Layer ===
def apply_metacortex(df):
    """
    Module 7: Signal Confidence Meter
    Computes unified confidence score and applies hard rejection for low scores.
    """
    df["pattern_match_score"] = df["pattern_match_score"].fillna(0.0)
    df["trap_map_score"] = df["trap_map_score"].fillna(0.0)
    df["mvrf_reject"] = df["mvrf_reject"].fillna(False)

    df["confidence_score"] = (
        0.35 * df["pattern_match_score"] +
        0.25 * df["trap_map_score"] +
        0.20 * (1.0 - df["mvrf_reject"].astype(float))
    ).round(2)

    df["signal_confidence"] = df["confidence_score"]
    df["signal_accepted"] = df["signal_confidence"] >= 0.25

    total_signals = len(df[df["entry_signal"] == 1])
    accepted_signals = df["signal_accepted"].sum()

    logging.info(f"Metacortex filtered: {accepted_signals} accepted / {total_signals} entry signals.")

    return df


import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")



def safe_str(val, fmt=None):
    if val is None:
        return "None"
    if fmt:
        try:
            return fmt.format(val)
        except Exception:
            return str(val)
    return str(val)


def calculate_volume(entry_price, sl_price, symbol, risk_pct=0.12):
    account_info = mt5.account_info()
    symbol_info = mt5.symbol_info(symbol)

    if account_info is None or symbol_info is None:
        logging.error(f"[{symbol}] Account info or symbol info unavailable. Cannot calculate volume.")
        return None

    balance = account_info.balance
    risk_amount = balance * risk_pct

    point = symbol_info.point  # expected 0.001
    digits = symbol_info.digits  # expected 3
    sl_distance = abs(entry_price - sl_price)

    logging.info(f"[{symbol}] Entry price: {entry_price}, SL price: {sl_price}, SL distance: {sl_distance:.5f}")

    if sl_distance < point * 0.1:
        logging.warning(f"[{symbol}] SL distance {sl_distance:.5f} too small to calculate volume.")
        return None

    tick_value = symbol_info.trade_tick_value  # ~0.67655
    tick_size = symbol_info.trade_tick_size  # 0.001
    contract_size = symbol_info.trade_contract_size  # 100000.0

    logging.info(f"[{symbol}] Tick value: {tick_value}, Tick size: {tick_size}, Contract size: {contract_size}")

    if tick_value <= 0 or tick_size <= 0 or contract_size <= 0:
        logging.error(f"[{symbol}] Invalid tick or contract info (tick_value={tick_value}, tick_size={tick_size}, contract_size={contract_size}).")
        return None

    # Calculate pip value per lot
    pip_value_per_lot = (tick_value / tick_size) * point
    risk_per_lot = (sl_distance / point) * pip_value_per_lot

    logging.info(f"[{symbol}] Pip value per lot: {pip_value_per_lot:.5f}, Risk per lot: {risk_per_lot:.5f}")

    if risk_per_lot <= 0:
        logging.error(f"[{symbol}] Calculated risk per lot is zero or negative: {risk_per_lot}.")
        return None

    raw_volume = risk_amount / risk_per_lot

    step = symbol_info.volume_step if symbol_info.volume_step else 0.01
    volume = round(raw_volume / step) * step

    logging.info(f"[{symbol}] Raw volume: {raw_volume:.5f}, Rounded volume (step {step}): {volume}")

    if volume < symbol_info.volume_min:
        logging.warning(f"[{symbol}] Calculated volume {volume} below broker minimum {symbol_info.volume_min}. Trade skipped.")
        return None

    volume = min(volume, symbol_info.volume_max)

    volume = round(volume, 2)  # round to 2 decimals for safety

    logging.info(f"[{symbol}] Final calculated volume: {volume} lots for risk_pct={risk_pct*100}% with SL distance {sl_distance:.5f}.")
    logging.info(f"[{symbol}] Final calculated volume: {volume} lots")

    return volume

from datetime import datetime, timedelta
import time
import logging
import MetaTrader5 as mt5

# Store all processed signals for this session
processed_signals = []

import pandas as pd
import logging

import logging
import time
from datetime import datetime
import pandas as pd
import traceback

def safe_float(val, fallback=None):
    if val is None:
        return fallback
    try:
        return float(val)
    except Exception:
        return fallback

def check_signal_outcome(df, sig):
    """
    Determine if a signal hit TP or SL first, ignoring the 'direction' label.
    Works purely with numeric TP and SL values relative to the price.
    """
    if sig.get("outcome") is not None:
        return sig  # already calculated

    # Find the signal index
    signal_idx = None
    sig_time = sig.get("time")
    if sig_time and "time" in df.columns:
        times = pd.to_datetime(df["time"])
        sig_ts = pd.to_datetime(sig_time)
        earlier = times[times <= sig_ts]
        if not earlier.empty:
            signal_idx = earlier.index[-1]

    if signal_idx is None:
        signal_idx = sig.get("candle_index")
        if signal_idx is None or not (0 <= signal_idx < len(df)):
            return sig  # cannot find signal

    sl = float(sig.get("sl", None))
    tp = float(sig.get("tp", None))
    entry = float(sig.get("entry", df.iloc[signal_idx]["close"]))

    if sl is None or tp is None:
        return sig  # cannot determine outcome without SL/TP

    # Iterate subsequent candles
    for i in range(signal_idx + 1, len(df)):
        row = df.iloc[i]
        high, low, open_ = row["high"], row["low"], row["open"]

        # If both TP and SL hit in same candle
        tp_hit = (tp >= entry and high >= tp) or (tp < entry and low <= tp)
        sl_hit = (sl >= entry and high >= sl) or (sl < entry and low <= sl)

        if tp_hit and sl_hit:
            # Compare distance from open to determine which hit first
            dist_tp = abs(open_ - tp)
            dist_sl = abs(open_ - sl)
            sig["outcome"] = "win" if dist_tp <= dist_sl else "loss"
            return sig
        elif tp_hit:
            sig["outcome"] = "win"
            return sig
        elif sl_hit:
            sig["outcome"] = "loss"
            return sig

    # If neither TP nor SL hit
    sig["outcome"] = "open"
    return sig


def fetch_data(symbol):
    """
    Fetch M15 candles with robust incremental update:
    - First call: fetch 128,000 candles
    - Subsequent calls: fill any missing candles and append the latest
    """
    import pandas as pd
    import logging
    import MetaTrader5 as mt5
    from datetime import timedelta

    # --- Initialize memory storage ---
    if not hasattr(fetch_data, "_cache"):
        fetch_data._cache = {}
    if symbol not in fetch_data._cache:
        fetch_data._cache[symbol] = pd.DataFrame()

    cached_df = fetch_data._cache[symbol]

    # --- First fetch (full history) ---
    if cached_df.empty:
        logging.info(f"{symbol} — Fetching initial 8,000 candles")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0,8000)
        if rates is None or len(rates) == 0:
            logging.error(f"{symbol} — Failed to fetch initial rates")
            return None

        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["time"] = df["time"].dt.tz_convert("America/Jamaica")

        fetch_data._cache[symbol] = df
        cached_df = df.copy()

    else:
        # --- Incremental fetch with gap-fill ---
        last_time = cached_df["time"].max()
        next_expected_time = last_time + timedelta(minutes=15)

        now = pd.Timestamp.now(tz="America/Jamaica")
        # Compute how many M15 candles are missing (failsafe)
        missing_candles = int((now - next_expected_time) / timedelta(minutes=15))
        if missing_candles < 1:
            missing_candles = 1  # always fetch at least the latest

        logging.info(f"{symbol} — Fetching {missing_candles} candle(s) after {last_time}")

        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M15, next_expected_time.to_pydatetime(), missing_candles)
        if rates is not None and len(rates) > 0:
            df_new = pd.DataFrame(rates)
            df_new.rename(columns={"tick_volume": "volume"}, inplace=True)
            df_new["time"] = pd.to_datetime(df_new["time"], unit="s", utc=True)
            df_new["time"] = df_new["time"].dt.tz_convert("America/Jamaica")

            # Remove duplicates if overlapping
            df_new = df_new[~df_new["time"].isin(cached_df["time"])]

            if not df_new.empty:
                fetch_data._cache[symbol] = pd.concat([cached_df, df_new], ignore_index=True)
                cached_df = fetch_data._cache[symbol]

    # --- Print last 10 candles ---
    print(cached_df.tail(10))

    # --- Save to CSV ---
    cached_df.to_csv(f"{symbol.lower()}_5m_data.csv", index=False)

    return cached_df

def update_win_rate():
    total_signals = len(processed_signals)
    closed = [s for s in processed_signals if s.get("outcome") in ("win", "loss")]
    total_closed = len(closed)
    wins = sum(1 for s in closed if s.get("outcome") == "win")

    closed_win_rate = round((wins / total_closed) * 100, 2) if total_closed > 0 else None
    all_win_rate = round((wins / total_signals) * 100, 2) if total_signals > 0 else None

    logging.info(
        f"Signals total={total_signals} | Closed={total_closed} | Wins={wins} | "
        f"Closed Win Rate={closed_win_rate if closed_win_rate is not None else 'N/A'}% | "
        f"Wins/Total={all_win_rate if all_win_rate is not None else 'N/A'}%"
    )
    return closed_win_rate
import pandas as pd
import numpy as np
import logging

import pandas as pd


def analyze_losing_trade_attributes(df, signals_df, min_trades=100):
    """
    Enhanced losing trade analysis:
    - Candle-level attribute comparison (body, wick, range, bull/bear ratio, volume)
    - Loss type classification: 'wicked' or 'direction'
    - Pattern-specific suggested risk/reward ratios
    - What-if filter simulation to propose trade filters
    """
    import numpy as np
    import pandas as pd
    from itertools import combinations, chain

    eps = 1e-9

    # --- Ensure candle-level attributes exist ---
    if 'body_size' not in df.columns:
        df['body_size'] = (df['close'] - df['open']).abs()
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['bull_bear_ratio'] = (df['close'] - df['open']) / (df['candle_range'] + eps)
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(10).mean() + eps)

    # --- Merge signals with candle attributes ---
    signals_df = signals_df.merge(
        df[['body_size', 'upper_wick', 'lower_wick', 'candle_range',
            'bull_bear_ratio', 'volume_ratio', 'high', 'low']],
        left_index=True, right_index=True, how='left'
    )

    # --- Classify losing trades ---
    losing = signals_df[signals_df['outcome'] == 'loss'].copy()
    winning = signals_df[signals_df['outcome'] == 'win'].copy()

    if losing.empty:
        print("⚠️ No losing trades found for analysis.")
        return None, None

    # Loss type: wicked out vs directional failure
    def classify_loss(row):
        if row['direction'] == 'buy':
            return 'wicked' if row['low'] < row['sl'] else 'direction'
        elif row['direction'] == 'sell':
            return 'wicked' if row['high'] > row['sl'] else 'direction'
        else:
            return 'unknown'

    losing['loss_type'] = losing.apply(classify_loss, axis=1)

    # --- Compute pattern-specific suggested R ---
    pattern_rr_suggestion = {}
    for pattern, group in signals_df.groupby('pattern_id'):
        wins = group[group['outcome'] == 'win']
        if not wins.empty:
            avg_sl = (wins['entry'] - wins['sl']).abs().mean()
            avg_tp = (wins['tp'] - wins['entry']).abs().mean()
            pattern_rr_suggestion[pattern] = round(avg_tp / (avg_sl + eps), 2)
        else:
            pattern_rr_suggestion[pattern] = 1.0  # fallback

    signals_df['pattern_rr'] = signals_df['pattern_id'].map(pattern_rr_suggestion)

    # --- Pattern-level win/loss stats ---
    pattern_stats = signals_df.groupby('pattern_id')['outcome'].value_counts().unstack(fill_value=0)
    pattern_stats['total'] = pattern_stats.sum(axis=1)
    pattern_stats['winrate'] = (pattern_stats.get('win', 0) / pattern_stats['total'] * 100).round(2)
    pattern_stats['suggested_R'] = pattern_stats.index.map(pattern_rr_suggestion)

    print("\n📊 Pattern-Level Win/Loss Stats with Suggested R:")
    print(pattern_stats.sort_values('winrate', ascending=False))

    # --- Compute attribute stats ---
    stats = {}
    for col in ['body_size', 'upper_wick', 'lower_wick',
                'candle_range', 'bull_bear_ratio', 'volume_ratio']:
        stats[col] = {
            'losing_mean': losing[col].mean(),
            'winning_mean': winning[col].mean(),
            'losing_median': losing[col].median(),
            'winning_median': winning[col].median()
        }

    stats_df = pd.DataFrame(stats).T
    print("\n🔎 Attribute Comparison (Losing vs Winning Trades):")
    print(stats_df)

    # --- Propose filter rules ---
    proposals = {}
    for col in stats_df.index:
        if stats_df.loc[col, 'losing_mean'] > stats_df.loc[col, 'winning_mean']:
            proposals[col] = ('high', stats_df.loc[col, 'losing_mean'])
        else:
            proposals[col] = ('low', stats_df.loc[col, 'losing_mean'])

    print("\n📌 Proposed Filter Ideas:")
    for col, (direction, val) in proposals.items():
        print(f"- Filter out trades where {col} is {direction} (losing mean={val:.6f})")

    # --- Generate all filter combinations ---
    all_cols = list(proposals.keys())
    best_combo = None
    best_winrate = 0
    best_filtered_df = signals_df.copy()

    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    print("\n🧪 What-If Filter Simulation (All Combinations):")
    for combo in powerset(all_cols):
        df_combo = signals_df.copy()
        for col in combo:
            direction, val = proposals[col]
            if direction == 'high':
                df_combo = df_combo[df_combo[col] <= val]
            else:
                df_combo = df_combo[df_combo[col] >= val]

        wins = (df_combo['outcome'] == 'win').sum()
        total = len(df_combo)
        if total < min_trades:
            continue

        winrate = wins / total * 100
        print(f"Filter combo {combo} → {total} trades, winrate={winrate:.2f}%")

        if winrate > best_winrate:
            best_winrate = winrate
            best_combo = combo
            best_filtered_df = df_combo.copy()

    print("\n✅ Best Filter Combination:")
    if best_combo:
        print(f"Filters: {best_combo} → {len(best_filtered_df)} trades, winrate={best_winrate:.2f}%")
    else:
        print("No combination met the minimum trade requirement.")

    # --- Return upgraded signals with loss_type and suggested R ---
    return best_combo, best_filtered_df


if __name__ == "__main__":
    processed_signals = []
import MetaTrader5 as mt5
import pandas as pd
import logging
import time
import traceback
from datetime import datetime, timedelta

# --- Memory for last accepted count per symbol ---
last_accepted_count = {}

def get_last_closed_m15(now):
    """Return the datetime of the last closed M15 candle."""
    minute = (now.minute // 15) * 15
    last_closed = now.replace(minute=minute, second=0, microsecond=0)
    # if current time is exactly on the candle boundary, last candle closed 15 min ago
    if now.minute % 15 == 0 and now.second == 0:
        last_closed -= timedelta(minutes=15)
    return last_closed
def print_win_loss_sequence(processed_signals, last_n=5000):
    seq = [s["outcome"] for s in processed_signals if s.get("outcome") in ("win", "loss")]
    if not seq:
        logging.info("Win/Loss sequence: N/A")
        return
    tail = seq[-last_n:]
    logging.info(f"Win/Loss sequence (last {len(tail)}): {' '.join(tail)}")
def trading_loop():
    import time
    import logging
    from datetime import datetime
    import MetaTrader5 as mt5  # assuming MetaTrader5 is already imported

    symbols = ["USDJPY"]
    poll_interval = 1  # slightly higher to reduce CPU load

    logging.info("Starting Undecillion trading loop (timestamp-based + analytics)")

    SCRIPT_START_TIME = datetime.now()
    STARTUP_DELAY = timedelta(minutes=0.1)
    startup_aligned = False

    last_traded_signal_time = {s: None for s in symbols}

    # --- GLOBAL ANALYTICS MEMORY ---
    global processed_signals
    processed_signals = []

    # Track last seen candle globally
    last_seen_candle_time = None

    while True:
        now = datetime.now()

        if now - SCRIPT_START_TIME < STARTUP_DELAY:
            time.sleep(poll_interval)
            continue

        # =========================
        # STARTUP ALIGNMENT (ONCE)
        # =========================
        if not startup_aligned:
            for symbol in symbols:
                df = fetch_data(symbol)
                if df is None or df.empty:
                    continue

                # Drop the forming candle to avoid phantoms
                df = df.iloc[:-1]

                df = generate_core_signals(df)
                df = apply_trap_mapping(df)
                df = apply_liquidity_filter(df)
                df = undecillion_signal_pipeline(df)
                df = apply_mvrf_filter(df)
                df = undecillion_signal_pipeline_with_mvrf(df)
                df = apply_reclaim_layer(df)
                df = apply_metacortex(df)

                df = df.reset_index(drop=True)

                accepted = df[df.get("signal_accepted", False)]

                for _, row in accepted.iterrows():
                    processed_signals.append({
                        "time": row["time"],
                        "candle_index": row.name,
                        "entry": row.get("entry_price", row.get("entry")),
                        "sl": row.get("sl"),
                        "tp": row.get("tp"),
                        "direction": row.get("direction"),
                        "pattern_id": row.get("pattern_id", "unknown"),
                        "outcome": None
                    })

                if not accepted.empty:
                    last_traded_signal_time[symbol] = accepted["time"].iloc[-1]

                for sig in processed_signals:
                    check_signal_outcome(df, sig)

                update_win_rate()
                print_win_loss_sequence(processed_signals)

                logging.info(
                    f"{symbol} — startup alignment complete, "
                    f"last_traded_signal_time set to {last_traded_signal_time[symbol]}"
                )

            startup_aligned = True
            time.sleep(poll_interval)
            continue

        # =========================
        # LIVE LOOP WITH CANDLE GUARD
        # =========================
        for symbol in symbols:
            try:
                info = mt5.symbol_info(symbol)
                if not info or not getattr(info, "visible", True):
                    continue
                if not is_valid_session():
                    continue

                # -------------------------
                # FETCH RATES & NEW CANDLE DETECTION
                # -------------------------
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 3)
                current_candle_time = rates[-1]['time']

                # Skip if we already processed this candle
                if current_candle_time == last_seen_candle_time:
                    continue

                last_seen_candle_time = current_candle_time

                # Small delay to ensure data is fully updated
                time.sleep(2)

                df = fetch_data(symbol)
                if df is None or df.empty:
                    continue

                # Drop forming candle to avoid phantom triggers
                df = df.iloc[:-1]

                df = generate_core_signals(df)
                df = apply_trap_mapping(df)
                df = apply_liquidity_filter(df)
                df = undecillion_signal_pipeline(df)
                df = apply_mvrf_filter(df)
                df = undecillion_signal_pipeline_with_mvrf(df)
                df = apply_reclaim_layer(df)
                df = apply_metacortex(df)

                df = df.reset_index(drop=True)

                accepted = df[df.get("signal_accepted", False)]
                if not accepted.empty:
                    latest = accepted.iloc[-1]
                    sig_time = latest["time"]

                    if last_traded_signal_time[symbol] is None or sig_time > last_traded_signal_time[symbol]:

                        entry = latest.get("entry_price", latest.get("entry"))
                        sl = latest.get("sl")
                        tp = latest.get("tp")
                        direction = latest.get("direction")

                        try:
                            volume = calculate_volume(entry, sl, symbol, risk_pct=0.12)
                        except Exception:
                            volume = 0.01

                        try:
                            result = place_trade(
                                symbol, direction, entry, sl, tp,
                                volume, slippage=50, magic_number=20250708
                            )

                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                last_traded_signal_time[symbol] = sig_time
                                logging.info(f"{symbol} — TRADE CONFIRMED at {sig_time}")
                            else:
                                logging.error(
                                    f"{symbol} — TRADE REJECTED | "
                                    f"retcode={getattr(result, 'retcode', None)}"
                                )

                        except Exception:
                            logging.exception("Trade execution failed")

                for sig in processed_signals:
                    check_signal_outcome(df, sig)

                update_win_rate()
                print_win_loss_sequence(processed_signals)

            except Exception as e:
                logging.error(f"{symbol} — poll error: {e}")
                logging.error(traceback.format_exc())

        time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trading_loop()

