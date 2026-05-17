"""
app.py
======
FastAPI server that wraps predict.py into a REST API.

Endpoints:
----------
  GET  /              -> welcome message
  GET  /health        -> API + model health check
  GET  /predict       -> run live prediction right now
  GET  /history       -> last N predictions from log file
  GET  /history/stats -> summary stats of all past predictions

Usage:
------
  # Start the server:
  uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload

  # Call predict endpoint:
  curl http://localhost:8000/predict

  # View last 5 predictions:
  curl http://localhost:8000/history?limit=5

Author : Bitcoin ML Project
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src/ to path so we can import predict.py
# predict.py lives in src/, app.py lives in app/
# This adds the parent directory so both are reachable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, '..', 'src')
sys.path.insert(0, SRC_DIR)

import predict as predictor   # imports predict.py functions

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

LOG_FILE   = os.path.join(BASE_DIR, '..', 'logs', 'predictions.json')
MODEL_DIR  = os.path.join(BASE_DIR, '..', 'models')

# -----------------------------------------------------------------------------
# FASTAPI APP SETUP
# -----------------------------------------------------------------------------

app = FastAPI(
    title       = 'Bitcoin ML Prediction API',
    description = (
        'Real-time Bitcoin price direction and price prediction '
        'using XGBoost trained on 5-minute OHLCV candles from Binance.'
    ),
    version     = '1.0.0',
    docs_url    = '/docs',    # Swagger UI at /docs
    redoc_url   = '/redoc',   # ReDoc UI  at /redoc
)

# CORS — allows any frontend (React, Vue etc.) to call this API
# In production you would restrict origins to your actual domain
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ['*'],
    allow_credentials = True,
    allow_methods     = ['*'],
    allow_headers     = ['*'],
)

# -----------------------------------------------------------------------------
# PYDANTIC RESPONSE MODELS
# These define the exact shape of each API response.
# FastAPI uses them for validation and auto-documentation.
# -----------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Schema for a single prediction result."""
    timestamp         : str
    current_price     : float
    predicted_price   : float
    price_change_est  : float
    pct_change_est    : float
    direction         : str       # "UP" or "DOWN"
    confidence        : float     # 0.0 to 1.0
    signal_strength   : str       # "STRONG", "MODERATE", "WEAK"
    trustworthy       : bool      # confidence >= threshold
    threshold_used    : float
    fetched_at        : str       # when the API call was made

class HealthResponse(BaseModel):
    """Schema for the health check endpoint."""
    status            : str       # "ok" or "degraded"
    api_version       : str
    models_loaded     : bool
    model_features    : int
    classifier_ready  : bool
    regressor_ready   : bool
    timestamp         : str

class HistoryResponse(BaseModel):
    """Schema for the history endpoint."""
    count             : int
    predictions       : list

class StatsResponse(BaseModel):
    """Schema for the history stats endpoint."""
    total_predictions : int
    up_predictions    : int
    down_predictions  : int
    up_pct            : float
    avg_confidence    : float
    avg_price_error   : Optional[float]
    trustworthy_pct   : float
    first_prediction  : Optional[str]
    last_prediction   : Optional[str]


# -----------------------------------------------------------------------------
# HELPER — load prediction history from log file
# -----------------------------------------------------------------------------

def load_history(limit: int = 100) -> list:
    """
    Reads the predictions.json log file and returns the last `limit` entries.

    The log file is in JSON Lines format (one JSON object per line).
    We read all lines and return the most recent ones.

    Returns empty list if file doesn't exist yet.
    """
    if not os.path.exists(LOG_FILE):
        return []

    predictions = []
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    predictions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue   # skip malformed lines

    # Return most recent `limit` predictions (newest last)
    return predictions[-limit:]


# -----------------------------------------------------------------------------
# HELPER — check if models are available and loadable
# -----------------------------------------------------------------------------

def check_models_health() -> dict:
    """
    Tries to load models and returns a health status dict.
    Used by the /health endpoint.
    """
    required_files = [
        'xgb_classifier.pkl',
        'xgb_regressor.pkl',
        'scaler.pkl',
        'feature_cols.pkl',
    ]

    # Check all files exist
    all_exist = all(
        os.path.exists(os.path.join(MODEL_DIR, f))
        for f in required_files
    )

    if not all_exist:
        return {
            'models_loaded'    : False,
            'classifier_ready' : False,
            'regressor_ready'  : False,
            'model_features'   : 0,
        }

    try:
        models = predictor.load_models(MODEL_DIR)
        return {
            'models_loaded'    : True,
            'classifier_ready' : models['clf'] is not None,
            'regressor_ready'  : models['reg'] is not None,
            'model_features'   : len(models['feature_cols']),
        }
    except Exception as e:
        log.error(f'Model health check failed: {e}')
        return {
            'models_loaded'    : False,
            'classifier_ready' : False,
            'regressor_ready'  : False,
            'model_features'   : 0,
        }


# -----------------------------------------------------------------------------
# ENDPOINT 1 — Root / Welcome
# -----------------------------------------------------------------------------

@app.get('/', response_class=HTMLResponse, tags=['General'])
def root():
    """
    Serves the interactive Bitcoin prediction dashboard.
    Opens automatically when you visit your Render URL.
    """
    html_path = os.path.join(BASE_DIR, 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    # Fallback JSON if HTML not found
    return HTMLResponse(content="""
    <html><body style="font-family:monospace;padding:2rem;background:#0d0d0d;color:#f1f1f1">
    <h2>BTC Prediction API</h2>
    <p>Endpoints: <a href="/docs" style="color:#F7931A">/docs</a> &nbsp;
    <a href="/predict" style="color:#F7931A">/predict</a> &nbsp;
    <a href="/health" style="color:#F7931A">/health</a></p>
    </body></html>
    """)


# -----------------------------------------------------------------------------
# ENDPOINT 2 — Health Check
# -----------------------------------------------------------------------------

@app.get('/health', response_model=HealthResponse, tags=['General'])
def health():
    """
    Health check — confirms API is running and models are loaded.

    Use this to monitor if the service is alive.
    A load balancer or Render health check should call this endpoint.

    Returns:
        status: "ok" if all models loaded, "degraded" if something is wrong
    """
    model_health = check_models_health()

    all_ok = (
        model_health['models_loaded'] and
        model_health['classifier_ready'] and
        model_health['regressor_ready']
    )

    return HealthResponse(
        status           = 'ok' if all_ok else 'degraded',
        api_version      = '1.0.0',
        timestamp        = datetime.now(timezone.utc).isoformat(),
        **model_health
    )


# -----------------------------------------------------------------------------
# ENDPOINT 3 — Live Prediction (the main endpoint)
# -----------------------------------------------------------------------------

@app.get('/predict', response_model=PredictionResponse, tags=['Prediction'])
def get_prediction():
    """
    Fetches live BTC data from Binance, runs both ML models,
    and returns a prediction for the next 5-minute candle.

    This endpoint:
    1. Hits Binance API to get latest 200 candles
    2. Computes 66 technical indicators
    3. Runs XGBoost classifier -> direction + confidence
    4. Runs XGBoost regressor  -> predicted price
    5. Returns the full prediction result

    Response fields:
    - **current_price**   : latest BTC close price
    - **predicted_price** : model's prediction for next candle close
    - **direction**       : "UP" or "DOWN"
    - **confidence**      : probability of UP (0.0 to 1.0)
    - **signal_strength** : "STRONG" / "MODERATE" / "WEAK"
    - **trustworthy**     : True if confidence >= threshold (0.55)

    Note: Each call hits Binance API and runs inference (~1-2 seconds).
    """
    try:
        log.info('POST /predict - running live prediction')
        result = predictor.run_prediction()

        # Add the API call timestamp
        result['fetched_at'] = datetime.now(timezone.utc).isoformat()

        return PredictionResponse(**result)

    except FileNotFoundError as e:
        # Models not trained yet
        raise HTTPException(
            status_code = 503,
            detail      = (
                f'Model files not found: {str(e)}. '
                f'Please run train_model.ipynb to train the models first.'
            )
        )

    except Exception as e:
        # Binance API failure, feature error, etc.
        log.error(f'/predict failed: {e}')
        raise HTTPException(
            status_code = 500,
            detail      = f'Prediction failed: {str(e)}'
        )


# -----------------------------------------------------------------------------
# ENDPOINT 4 — Prediction History
# -----------------------------------------------------------------------------

@app.get('/history', response_model=HistoryResponse, tags=['History'])
def get_history(
    limit: int = Query(
        default = 10,
        ge      = 1,
        le      = 500,
        description = 'Number of recent predictions to return (1-500)'
    )
):
    """
    Returns the last N predictions from the prediction log file.

    Every call to /predict is automatically saved to logs/predictions.json.
    This endpoint reads that file and returns recent history.

    Use this to:
    - See how the model has been performing recently
    - Track direction accuracy over time
    - Build a frontend chart of predictions vs actual

    Parameters:
    - **limit**: how many recent predictions to return (default 10, max 500)
    """
    try:
        predictions = load_history(limit=limit)
        return HistoryResponse(
            count       = len(predictions),
            predictions = predictions
        )
    except Exception as e:
        log.error(f'/history failed: {e}')
        raise HTTPException(
            status_code = 500,
            detail      = f'Could not load history: {str(e)}'
        )


# -----------------------------------------------------------------------------
# ENDPOINT 5 — History Stats
# -----------------------------------------------------------------------------

@app.get('/history/stats', response_model=StatsResponse, tags=['History'])
def get_history_stats():
    """
    Returns summary statistics across all saved predictions.

    Useful for evaluating how the model is performing in production:
    - What % of predictions say UP vs DOWN?
    - What is the average confidence score?
    - What % of predictions pass the confidence threshold?

    Note: price error is only meaningful if you have ground truth data
    (i.e., you waited for the next candle to close and compared).
    """
    try:
        all_predictions = load_history(limit=10000)   # load all

        if not all_predictions:
            return StatsResponse(
                total_predictions = 0,
                up_predictions    = 0,
                down_predictions  = 0,
                up_pct            = 0.0,
                avg_confidence    = 0.0,
                avg_price_error   = None,
                trustworthy_pct   = 0.0,
                first_prediction  = None,
                last_prediction   = None,
            )

        df = pd.DataFrame(all_predictions)

        up_count   = (df['direction'] == 'UP').sum()
        down_count = (df['direction'] == 'DOWN').sum()
        total      = len(df)

        # Average confidence
        avg_conf = float(df['confidence'].mean()) if 'confidence' in df else 0.0

        # Trustworthy %
        trust_pct = float(
            (df['trustworthy'].sum() / total * 100)
            if 'trustworthy' in df else 0.0
        )

        # Average price error (if available)
        avg_price_err = None
        if 'price_change_est' in df:
            avg_price_err = float(df['price_change_est'].abs().mean())

        return StatsResponse(
            total_predictions = total,
            up_predictions    = int(up_count),
            down_predictions  = int(down_count),
            up_pct            = round(up_count / total * 100, 2),
            avg_confidence    = round(avg_conf, 4),
            avg_price_error   = round(avg_price_err, 2) if avg_price_err else None,
            trustworthy_pct   = round(trust_pct, 2),
            first_prediction  = df['timestamp'].iloc[0]  if 'timestamp' in df else None,
            last_prediction   = df['timestamp'].iloc[-1] if 'timestamp' in df else None,
        )

    except Exception as e:
        log.error(f'/history/stats failed: {e}')
        raise HTTPException(
            status_code = 500,
            detail      = f'Could not compute stats: {str(e)}'
        )


# -----------------------------------------------------------------------------
# STARTUP EVENT — runs once when server starts
# -----------------------------------------------------------------------------

@app.on_event('startup')
async def startup_event():
    """
    Runs when the FastAPI server starts up.

    We pre-check models are loaded and log a startup message.
    This way if models are missing, you know immediately on startup
    rather than discovering it when the first /predict call comes in.
    """
    log.info('=' * 50)
    log.info('  Bitcoin ML Prediction API starting...')
    log.info('=' * 50)

    model_health = check_models_health()

    if model_health['models_loaded']:
        log.info(
            f"Models loaded OK | "
            f"Features: {model_health['model_features']}"
        )
    else:
        log.warning(
            'Models NOT found in models/ directory. '
            'Run train_model.ipynb before calling /predict'
        )

    log.info('API ready at http://0.0.0.0:8000')
    log.info('Docs available at http://0.0.0.0:8000/docs')


# -----------------------------------------------------------------------------
# SHUTDOWN EVENT — runs once when server stops
# -----------------------------------------------------------------------------

@app.on_event('shutdown')
async def shutdown_event():
    log.info('Bitcoin ML API shutting down.')


# -----------------------------------------------------------------------------
# RUN DIRECTLY
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'app:app',
        host     = '0.0.0.0',
        port     = 8000,
        reload   = True,    # auto-restart on code changes (dev mode)
        log_level= 'info'
    )
