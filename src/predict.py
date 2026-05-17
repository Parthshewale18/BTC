"""
predict.py
==========
Real-time Bitcoin prediction script.

What this file does:
--------------------
Every 5 minutes it:
  1. Fetches the latest 200 BTC/USDT candles from Binance
  2. Computes all technical indicators (same as training)
  3. Loads the saved XGBoost models
  4. Runs prediction
  5. Outputs:
       - predicted_price   -> exact next close price
       - direction         -> "UP" or "DOWN"
       - confidence        -> probability (0.0 to 1.0)

Usage:
------
  # Run once (single prediction):
  python src/predict.py

  # Run continuously every 5 minutes:
  python src/predict.py --loop

  # Custom confidence threshold:
  python src/predict.py --loop --threshold 0.60

Author : Bitcoin ML Project
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import json
import logging
import argparse
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import requests

from ta.trend      import EMAIndicator, MACD
from ta.momentum   import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume     import OnBalanceVolumeIndicator

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# Logs go to both console AND a file so you can review history later
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s | %(levelname)s | %(message)s',
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, 'predict.log'))
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SYMBOL    = 'BTCUSDT'
INTERVAL  = '5m'

# We fetch 200 candles so all indicators have enough warm-up period.
# EMA50 needs 50, MACD needs 35, rolling_30 needs 30 -> 200 is safe.
FETCH_LIMIT = 200

BASE_URL = 'https://api1.binance.com/api/v3/klines'

MODEL_DIR  = os.path.join(BASE_DIR, '..', 'models')
LOG_FILE   = os.path.join(BASE_DIR, '..', 'logs', 'predictions.json')

# Only trust predictions above this confidence level.
# Below this = model is uncertain, treat output with caution.
CONFIDENCE_THRESHOLD = 0.55

# Seconds between predictions in loop mode (5 minutes = one candle)
PREDICTION_INTERVAL = 300

# Binance column names (12 fields per candle)
COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'num_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Fetch latest candles from Binance
# ─────────────────────────────────────────────────────────────────────────────

def fetch_latest_candles(symbol=SYMBOL, interval=INTERVAL, limit=FETCH_LIMIT):
    """
    Fetches the most recent `limit` candles from Binance public API.

    No API key needed — Binance klines endpoint is public.
    Returns a clean DataFrame with datetime index and float columns.

    Why 200 candles?
    ----------------
    Technical indicators need history to warm up:
      EMA 50      -> needs 49 prior candles
      MACD(26,12) -> needs 25 prior candles
      Rolling 30  -> needs 29 prior candles
    200 gives us plenty of buffer. We only predict on the LAST row.
    """
    params   = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()

    df = pd.DataFrame(raw, columns=COLUMNS)

    # ms timestamps -> UTC datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)

    # Binance returns all values as strings -> cast to float
    num_cols = ['open', 'high', 'low', 'close', 'volume',
                'quote_asset_volume', 'taker_buy_base', 'taker_buy_quote']
    df[num_cols]     = df[num_cols].astype(float)
    df['num_trades'] = df['num_trades'].astype(int)

    df.drop(columns=['close_time', 'ignore'], inplace=True)
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)

    log.info(
        f"Fetched {len(df)} candles | "
        f"Latest close: ${df['close'].iloc[-1]:,.2f} | "
        f"Candle time : {df.index[-1]}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build features (must match training EXACTLY)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df):
    """
    Computes all 66 technical indicators on live candle data.

    CRITICAL RULE:
    Every single feature here must match EXACTLY what was computed in
    feature_engineering.ipynb during training. If training used EMA(9),
    we must compute EMA(9) here with the same parameters.

    Any mismatch = scaler sees different value ranges = garbage predictions.

    We compute features on all 200 candles for warm-up accuracy,
    but only the LAST row is used for the actual prediction.
    """
    fe = df.copy()

    # ── EMA (Exponential Moving Averages) ────────────────────────────────────
    fe['ema_9']  = EMAIndicator(close=fe['close'], window=9).ema_indicator()
    fe['ema_21'] = EMAIndicator(close=fe['close'], window=21).ema_indicator()
    fe['ema_50'] = EMAIndicator(close=fe['close'], window=50).ema_indicator()

    # Distance from EMA as % — how stretched is price from average?
    fe['price_vs_ema9']     = (fe['close'] - fe['ema_9'])  / fe['ema_9']  * 100
    fe['price_vs_ema21']    = (fe['close'] - fe['ema_21']) / fe['ema_21'] * 100
    fe['price_vs_ema50']    = (fe['close'] - fe['ema_50']) / fe['ema_50'] * 100

    # EMA crossover signals
    fe['ema9_above_ema21']  = (fe['ema_9']  > fe['ema_21']).astype(int)
    fe['ema21_above_ema50'] = (fe['ema_21'] > fe['ema_50']).astype(int)

    # ── RSI (Relative Strength Index) ────────────────────────────────────────
    fe['rsi_14'] = RSIIndicator(close=fe['close'], window=14).rsi()
    fe['rsi_7']  = RSIIndicator(close=fe['close'], window=7).rsi()
    fe['rsi_21'] = RSIIndicator(close=fe['close'], window=21).rsi()

    fe['rsi_overbought'] = (fe['rsi_14'] > 70).astype(int)
    fe['rsi_oversold']   = (fe['rsi_14'] < 30).astype(int)
    fe['rsi_slope']      = fe['rsi_14'].diff(3)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_ind = MACD(
        close       = fe['close'],
        window_slow = 26,
        window_fast = 12,
        window_sign = 9
    )
    fe['macd']              = macd_ind.macd()
    fe['macd_signal']       = macd_ind.macd_signal()
    fe['macd_hist']         = macd_ind.macd_diff()
    fe['macd_norm']         = fe['macd'] / fe['close'] * 100
    fe['macd_above_signal'] = (fe['macd'] > fe['macd_signal']).astype(int)
    fe['macd_hist_slope']   = fe['macd_hist'].diff(2)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = BollingerBands(close=fe['close'], window=20, window_dev=2)
    fe['bb_upper']       = bb.bollinger_hband()
    fe['bb_mid']      = bb.bollinger_mavg()
    fe['bb_lower']       = bb.bollinger_lband()
    fe['bb_width']       = bb.bollinger_wband()
    fe['bb_pct']         = bb.bollinger_pband()
    fe['bb_above_upper'] = (fe['close'] > fe['bb_upper']).astype(int)
    fe['bb_below_lower'] = (fe['close'] < fe['bb_lower']).astype(int)

    # ── Volume features ───────────────────────────────────────────────────────
    fe['obv'] = OnBalanceVolumeIndicator(
        close=fe['close'], volume=fe['volume']
    ).on_balance_volume()
    fe['volume_ema_20'] = EMAIndicator(
        close=fe['volume'], window=20
    ).ema_indicator()
    fe['volume_ratio'] = fe['volume'] / fe['volume_ema_20']
    fe['high_volume']  = (fe['volume_ratio'] > 2.0).astype(int)

    # ── Lag features ─────────────────────────────────────────────────────────
    # shift(+N) looks BACKWARD in time — safe, no leakage
    for lag in range(1, 6):
        fe[f'close_lag_{lag}'] = fe['close'].shift(lag)

    fe['return_1']  = fe['close'].pct_change(1)  * 100
    fe['return_3']  = fe['close'].pct_change(3)  * 100
    fe['return_5']  = fe['close'].pct_change(5)  * 100
    fe['return_12'] = fe['close'].pct_change(12) * 100

    for lag in range(1, 4):
        fe[f'volume_lag_{lag}'] = fe['volume'].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    fe['rolling_mean_10'] = fe['close'].rolling(10).mean()
    fe['rolling_std_10']  = fe['close'].rolling(10).std()
    fe['rolling_min_10']  = fe['close'].rolling(10).min()
    fe['rolling_max_10']  = fe['close'].rolling(10).max()

    fe['rolling_mean_30'] = fe['close'].rolling(30).mean()
    fe['rolling_std_30']  = fe['close'].rolling(30).std()
    fe['rolling_min_30']  = fe['close'].rolling(30).min()
    fe['rolling_max_30']  = fe['close'].rolling(30).max()

    fe['zscore_10'] = (
        (fe['close'] - fe['rolling_mean_10']) / fe['rolling_std_10']
    )
    fe['zscore_30'] = (
        (fe['close'] - fe['rolling_mean_30']) / fe['rolling_std_30']
    )

    range_10 = (
        fe['rolling_max_10'] - fe['rolling_min_10']
    ).replace(0, np.nan)
    fe['price_position_10'] = (
        (fe['close'] - fe['rolling_min_10']) / range_10
    )

    # ── Candle shape features ─────────────────────────────────────────────────
    candle_range = (fe['high'] - fe['low']).replace(0, np.nan)

    fe['candle_body']       = (fe['close'] - fe['open']).abs()
    fe['body_ratio']        = fe['candle_body'] / candle_range
    fe['upper_wick']        = fe['high'] - fe[['open', 'close']].max(axis=1)
    fe['lower_wick']        = fe[['open', 'close']].min(axis=1) - fe['low']
    fe['upper_wick_ratio']  = fe['upper_wick']  / candle_range
    fe['lower_wick_ratio']  = fe['lower_wick']  / candle_range
    fe['is_bullish_candle'] = (fe['close'] > fe['open']).astype(int)

    return fe


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load saved models
# ─────────────────────────────────────────────────────────────────────────────

def load_models(model_dir=MODEL_DIR):
    """
    Loads the 4 files saved by train_model.ipynb:
      xgb_classifier.pkl  -> direction model
      xgb_regressor.pkl   -> price model
      scaler.pkl          -> StandardScaler (must match training)
      feature_cols.pkl    -> exact feature list in correct order

    Why reload every call?
    If you retrain and save new models, the prediction script picks
    them up automatically without needing a restart.
    """
    required = {
        'clf'         : 'xgb_classifier.pkl',
        'reg'         : 'xgb_regressor.pkl',
        'scaler'      : 'scaler.pkl',
        'feature_cols': 'feature_cols.pkl',
    }

    models = {}
    for key, filename in required.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nModel file not found: {path}\n"
                f"Run train_model.ipynb first to generate model files.\n"
                f"Expected location: {model_dir}/"
            )
        models[key] = joblib.load(path)

    log.info(
        f"Models loaded | "
        f"Features: {len(models['feature_cols'])} | "
        f"Dir: {model_dir}"
    )
    return models


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Run prediction on the latest candle
# ─────────────────────────────────────────────────────────────────────────────

def predict(models, feature_df):
    """
    Runs both models on the latest candle row and returns a prediction dict.

    Parameters
    ----------
    models     : dict from load_models()
    feature_df : DataFrame with all 66 indicator columns computed

    Returns
    -------
    dict with:
        timestamp        -> when this candle opened (UTC ISO format)
        current_price    -> current BTC close price
        predicted_price  -> model's prediction for next close
        price_change_est -> predicted_price - current_price
        pct_change_est   -> % change estimate
        direction        -> "UP" or "DOWN"
        confidence       -> probability of UP (0.0 to 1.0)
        signal_strength  -> "STRONG" / "MODERATE" / "WEAK"
        trustworthy      -> True if confidence >= threshold
    """
    clf          = models['clf']
    reg          = models['reg']
    scaler       = models['scaler']
    feature_cols = models['feature_cols']

    # Use only the last row — this is the most recent completed candle
    last_row = feature_df.iloc[[-1]]

    # ── Validate all required features exist ─────────────────────────────────
    missing = [c for c in feature_cols if c not in last_row.columns]
    if missing:
        raise ValueError(
            f"Missing features: {missing}\n"
            f"Check that build_features() matches feature_engineering.ipynb."
        )

    # ── Select features in exact training order ───────────────────────────────
    # The scaler expects columns in the same order it was fitted with.
    X = last_row[feature_cols]

    # ── Handle any NaN in the feature row ────────────────────────────────────
    nan_cols = X.columns[X.isnull().any()].tolist()
    if nan_cols:
        log.warning(f"NaN in features {nan_cols} — filling with 0")
        X = X.fillna(0)

    # ── Scale using the training scaler ──────────────────────────────────────
    X_scaled = scaler.transform(X)

    # ── Run classifier → direction + confidence ───────────────────────────────
    direction_pred = int(clf.predict(X_scaled)[0])
    confidence     = float(clf.predict_proba(X_scaled)[0][1])

    # ── Run regressor → next price ────────────────────────────────────────────
    predicted_price = float(reg.predict(X_scaled)[0])

    # ── Derived values ────────────────────────────────────────────────────────
    current_price    = float(feature_df['close'].iloc[-1])
    timestamp        = feature_df.index[-1]
    direction_label  = 'UP' if direction_pred == 1 else 'DOWN'
    price_change_est = predicted_price - current_price
    pct_change_est   = (price_change_est / current_price) * 100

    # Signal strength: how far is confidence from 0.5 (pure uncertainty)?
    # 0.50 = no idea,  0.65 = moderate,  0.80+ = strong
    dist = abs(confidence - 0.5)
    if dist >= 0.15:
        signal_strength = 'STRONG'
    elif dist >= 0.08:
        signal_strength = 'MODERATE'
    else:
        signal_strength = 'WEAK'

    trustworthy = confidence >= CONFIDENCE_THRESHOLD

    return {
        'timestamp'        : timestamp.isoformat(),
        'current_price'    : round(current_price,    2),
        'predicted_price'  : round(predicted_price,  2),
        'price_change_est' : round(price_change_est, 2),
        'pct_change_est'   : round(pct_change_est,   4),
        'direction'        : direction_label,
        'confidence'       : round(confidence,       4),
        'signal_strength'  : signal_strength,
        'trustworthy'      : trustworthy,
        'threshold_used'   : CONFIDENCE_THRESHOLD,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Append prediction to JSON log file
# ─────────────────────────────────────────────────────────────────────────────

def log_prediction(result, log_file=LOG_FILE):
    """
    Appends each prediction as one line to predictions.json.

    JSON Lines format — one JSON object per line.
    Load all history later with:
        df = pd.read_json('logs/predictions.json', lines=True)
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(json.dumps(result) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Pretty print to console
# ─────────────────────────────────────────────────────────────────────────────

def print_prediction(result):
    arrow  = 'UP ^' if result['direction'] == 'UP' else 'DOWN v'
    trust  = '[OK]' if result['trustworthy'] else '[LOW]'
    filled = int(result['confidence'] * 20)
    bar    = '#' * filled + '-' * (20 - filled)

    print()
    print('=' * 52)
    print('   BTC/USDT  --  Real-Time Prediction')
    print('=' * 52)
    print(f"  Time       : {result['timestamp'][:19]} UTC")
    print(f"  Current    : ${result['current_price']:>12,.2f}")
    print(f"  Predicted  : ${result['predicted_price']:>12,.2f}  ({result['pct_change_est']:+.3f}%)")
    print(f"  Direction  : {arrow}  ({result['signal_strength']})")
    print(f"  Confidence : [{bar}] {result['confidence']:.1%}")
    print(f"  Trustworthy: {trust}  (threshold: {result['threshold_used']:.0%})")
    print('=' * 52)

    if not result['trustworthy']:
        print('  WARNING: Low confidence -- treat with caution')
    print()
    
# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — ties all steps together
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction():
    """
    Runs one complete prediction cycle:
      fetch → features → load models → predict → log → print

    Returns the result dict so FastAPI (app.py) can also call this.
    """
    # 1. Fetch live candles from Binance
    log.info('--- Fetching live candles ---')
    raw_df = fetch_latest_candles()

    # 2. Compute all technical indicators
    log.info('--- Computing features ---')
    feature_df = build_features(raw_df)

    # Drop warm-up NaN rows (indicators need history to initialise)
    # We always keep at least the last row for prediction
    feature_df.dropna(inplace=True)

    if len(feature_df) == 0:
        raise ValueError(
            'All rows dropped after dropna — increase FETCH_LIMIT '
            f'(current: {FETCH_LIMIT})'
        )

    # 3. Load saved models
    log.info('--- Loading models ---')
    models = load_models()

    # 4. Predict
    log.info('--- Running prediction ---')
    result = predict(models, feature_df)

    # 5. Save to log file
    log_prediction(result)
    log.info(f"Logged to {LOG_FILE}")

    # 6. Print to console
    print_prediction(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LOOP — runs every 5 minutes aligned to candle close times
# ─────────────────────────────────────────────────────────────────────────────

def run_loop():
    """
    Runs predictions continuously, aligned to 5-minute candle boundaries.

    Why wait for candle close?
    Binance 5m candles close at :00, :05, :10, :15 ...
    We wait 10 seconds after each close to ensure Binance has
    processed the final price before we fetch it.

    Error handling:
    - Network errors  → wait 60s and retry
    - Unexpected errors → wait 30s and continue (don't crash)
    """
    log.info('Starting prediction loop — press Ctrl+C to stop')
    log.info(f'Confidence threshold : {CONFIDENCE_THRESHOLD}')
    log.info(f'Prediction interval  : every {PREDICTION_INTERVAL}s')

    while True:
        try:
            run_prediction()

            # Calculate sleep time until next candle close boundary
            now          = datetime.now(timezone.utc)
            elapsed_secs = (now.minute % 5) * 60 + now.second
            sleep_secs   = (PREDICTION_INTERVAL - elapsed_secs) + 10

            log.info(f'Next prediction in {sleep_secs}s ...')
            time.sleep(sleep_secs)

        except KeyboardInterrupt:
            log.info('Loop stopped by user.')
            break

        except requests.exceptions.RequestException as e:
            log.warning(f'Network error: {e} — retrying in 60s')
            time.sleep(60)

        except FileNotFoundError as e:
            # Models missing — no point retrying every 5 min
            log.error(str(e))
            log.error('Train the model first. Exiting.')
            break

        except Exception as e:
            log.error(f'Unexpected error: {e} — continuing in 30s')
            time.sleep(30)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Bitcoin Real-Time Prediction System'
    )
    parser.add_argument(
        '--loop',
        action  = 'store_true',
        help    = 'Run continuously every 5 min (default: run once and exit)'
    )
    parser.add_argument(
        '--threshold',
        type    = float,
        default = CONFIDENCE_THRESHOLD,
        help    = f'Confidence threshold 0.0-1.0 (default: {CONFIDENCE_THRESHOLD})'
    )
    args = parser.parse_args()

    # Apply CLI threshold override
    CONFIDENCE_THRESHOLD = args.threshold

    print('=' * 52)
    print('  Bitcoin ML  —  Real-Time Prediction System')
    print('=' * 52)
    print(f'  Mode        : {"LOOP (every 5 min)" if args.loop else "SINGLE RUN"}')
    print(f'  Symbol      : {SYMBOL}')
    print(f'  Interval    : {INTERVAL} candles')
    print(f'  Fetch limit : {FETCH_LIMIT} candles')
    print(f'  Threshold   : {CONFIDENCE_THRESHOLD}')
    print(f'  Model dir   : {MODEL_DIR}')
    print(f'  Log file    : {LOG_FILE}')
    print('=' * 52)
    print()

    if args.loop:
        run_loop()
    else:
        run_prediction()
