import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone


SYMBOL        = "BTCUSDT"
INTERVAL      = "5m"
LIMIT         = 1000          # max per Binance request
TOTAL_CANDLES = 50_000        # 50,000 x 5 min ≈ 173 days

BASE_URL    = "https://api.binance.com/api/v3/klines"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "btc_5m.csv")

COLUMNS = [
    "open_time",
    "open", "high", "low", "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — fetch one batch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_candles(symbol, interval, end_time=None, limit=1000):
    """
    Fetch one batch of candles from Binance.

    Parameters
    ----------
    end_time : Unix ms timestamp.
               Binance returns `limit` candles whose open_time < end_time.
               This is the KEY parameter for walking backwards in time.
               If None → returns the most recent `limit` candles.
    """

    params = {
        "symbol":   symbol,
        "interval": interval,
        "limit":    limit,
    }

    if end_time is not None:
        params["endTime"] = end_time

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — raw list → clean DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def raw_to_dataframe(raw_candles):
    """Convert Binance raw list-of-lists into a clean pandas DataFrame."""

    df = pd.DataFrame(raw_candles, columns=COLUMNS)

    # ms timestamp → UTC datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # strings → float
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote"
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"]  = df["num_trades"].astype(int)

    # drop unneeded columns
    df.drop(columns=["close_time", "ignore"], inplace=True)

    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — paginate backwards to collect TOTAL_CANDLES
# ─────────────────────────────────────────────────────────────────────────────

def download_historical_data(symbol=SYMBOL, interval=INTERVAL,
                              total_candles=TOTAL_CANDLES, limit=LIMIT):
    """
    Walk backwards in time using endTime pagination.

    How it works (step by step)
    ---------------------------
    Iteration 1:  endTime = None
                  → Binance returns latest 1000 candles  (T-999 … T)
                  → oldest candle in batch = T-999
                  → next endTime = open_time of T-999 (in ms)

    Iteration 2:  endTime = open_time(T-999)
                  → Binance returns 1000 candles ending BEFORE T-999
                  → covers  T-1999 … T-1000
                  → oldest candle = T-1999
                  → next endTime = open_time(T-1999)

    Iteration 3 … 50: same pattern — each batch goes 1000 candles further back

    Result: 50 × 1000 = 50,000 candles in chronological order
    """

    print("=" * 60)
    print(f"  Downloading {total_candles:,} candles | {symbol} {interval}")
    print("=" * 60)

    all_frames        = []
    candles_collected = 0
    end_time          = None          # first call → no endTime → gets latest
    num_requests      = (total_candles + limit - 1) // limit   # ceil division

    for i in range(num_requests):

        raw = fetch_candles(symbol, interval, end_time=end_time, limit=limit)

        if not raw:
            print("  ⚠️  Binance returned empty batch — stopping early.")
            break

        batch = raw_to_dataframe(raw)
        all_frames.append(batch)
        candles_collected += len(batch)

        # ── KEY: set next endTime to just BEFORE oldest candle in this batch ──
        # open_time of oldest candle  →  milliseconds  →  subtract 1 ms
        # This ensures the next batch ends strictly before this batch starts.
        oldest_open_time_ms = int(batch.index.min().timestamp() * 1000)
        end_time            = oldest_open_time_ms   # Binance endTime is exclusive

        # progress log
        oldest_str = batch.index.min().strftime("%Y-%m-%d %H:%M")
        newest_str = batch.index.max().strftime("%Y-%m-%d %H:%M")
        print(f"  Batch {i+1:>3}/{num_requests}  |  "
              f"{oldest_str} → {newest_str}  |  "
              f"Total so far: {candles_collected:,}")

        time.sleep(0.1)   # respect Binance rate limits

    # ── Combine, deduplicate, sort ─────────────────────────────────────────
    if not all_frames:
        raise RuntimeError("No data downloaded — check internet connection.")

    df = pd.concat(all_frames)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    df = df.iloc[-total_candles:]     # trim to exact count if we got extra

    print()
    print(f"  ✅ Done!  {len(df):,} candles collected.")
    print(f"  Date range : {df.index.min()}  →  {df.index.max()}")
    print(f"  Columns    : {list(df.columns)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — save to CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_data(df, filepath=OUTPUT_FILE):
    """Save DataFrame to CSV, creating directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_csv(filepath)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  💾 Saved → {filepath}  ({size_kb:.0f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — fetch latest N candles (used by predict.py at runtime)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_latest_candles(symbol=SYMBOL, interval=INTERVAL, limit=100):
    """
    Fetch the most recent `limit` candles.
    Called every 5 minutes by predict.py for live predictions.
    100 candles is enough to compute all technical indicators.
    """
    raw = fetch_candles(symbol, interval, end_time=None, limit=limit)
    df  = raw_to_dataframe(raw)
    print(f"  🔴 Live: {len(df)} candles  |  "
          f"Latest close = ${df['close'].iloc[-1]:,.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = download_historical_data()
    save_data(df)

    print("\nFirst 3 rows (oldest):")
    print(df.head(3)[["open","high","low","close","volume"]].to_string())

    print("\nLast 3 rows (newest):")
    print(df.tail(3)[["open","high","low","close","volume"]].to_string())

    print("\nStatistics:")
    print(df[["open","high","low","close","volume"]].describe().to_string())