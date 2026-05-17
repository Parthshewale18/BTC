# Bitcoin Real-Time Prediction System 🪙

A production-ready machine learning system that predicts Bitcoin price direction and next candle price using 5-minute OHLCV data from Binance.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Render](https://img.shields.io/badge/Deploy-Render-purple.svg)](https://render.com)

---

## What This System Does

Every 5 minutes it:
1. Fetches the latest BTC/USDT candles from Binance (no API key needed)
2. Computes 66 technical indicators (RSI, MACD, EMA, Bollinger Bands, etc.)
3. Runs two XGBoost models
4. Outputs a structured prediction:

```json
{
  "timestamp"        : "2026-05-16T10:35:00+00:00",
  "current_price"    : 79039.48,
  "predicted_price"  : 79105.22,
  "price_change_est" : 65.74,
  "pct_change_est"   : 0.0832,
  "direction"        : "UP",
  "confidence"       : 0.6124,
  "signal_strength"  : "MODERATE",
  "trustworthy"      : true,
  "threshold_used"   : 0.55
}
```

---

## System Architecture

```
Binance API (public, no key needed)
       │
       │  200 latest 5m candles
       ▼
┌─────────────────────────────────────────────┐
│              Data Pipeline                  │
│  data_collection.py  →  data_cleaning.ipynb │
│  feature_engineering.ipynb  (66 features)   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│              ML Models                      │
│  XGBoost Classifier  →  direction + confidence│
│  XGBoost Regressor   →  next close price    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│           FastAPI REST API                  │
│  GET /predict      →  live prediction       │
│  GET /health       →  model status          │
│  GET /history      →  past predictions      │
│  GET /history/stats→  performance summary   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
            Docker Container
            deployed on Render
```

---

## Project Structure

```
bitcoin-ml-project/
│
├── data/
│   ├── raw/                        ← downloaded OHLCV CSVs from Binance
│   └── processed/                  ← cleaned + feature-engineered data
│
│
├── src/
│   ├── data_collection.py          ← download historical BTC data
│   └── predict.py                  ← real-time prediction engine
│   ├── data_cleaning.ipynb         ← clean raw data, create targets
│   ├── feature_engineering.ipynb   ← build 66 technical indicators
│   └── train_model.ipynb           ← train, evaluate, backtest, save models
│
├── app/
│   └── app.py                      ← FastAPI REST API server
|   └── index.html 
│
├── models/                         ← saved trained model files
│   ├── xgb_classifier.pkl          ← direction prediction model
│   ├── xgb_regressor.pkl           ← price prediction model
│   ├── scaler.pkl                  ← fitted StandardScaler
│   └── feature_cols.pkl            ← feature column names (ordered)
│
├── logs/
│   └── predictions.jsonl           ← every prediction logged here
│
├── Dockerfile                      ← container build instructions
├── .dockerignore                   ← files excluded from Docker image
├── requirements.txt                ← pinned Python dependencies
└── README.md                       ← this file
```

---

## ML Models

### Task 1 — Direction Classification

```
Model   : XGBoost Classifier
Input   : 66 technical indicator features
Output  : 0 (DOWN) or 1 (UP)  +  confidence probability (0.0–1.0)
Metric  : Accuracy, F1-Score, ROC-AUC
```

### Task 2 — Price Regression

```
Model   : XGBoost Regressor
Input   : same 66 features
Output  : predicted next candle close price (USD)
Metric  : RMSE, MAE, MAPE
```

### Feature Groups (66 total)

| Group | Features | Count |
|-------|----------|-------|
| EMA | ema_9, ema_21, ema_50, crossovers, price distance | 8 |
| RSI | rsi_7, rsi_14, rsi_21, overbought, oversold, slope | 6 |
| MACD | macd, signal, histogram, normalised, crossover | 6 |
| Bollinger Bands | upper, lower, width, %B, breakout flags | 7 |
| Volume | OBV, volume EMA, volume ratio, high volume flag | 4 |
| Lag features | close_lag 1-5, volume_lag 1-3, returns 1/3/5/12 | 13 |
| Rolling stats | mean, std, min, max (10 & 30 periods), z-score | 9 |
| Candle shape | body, wicks, ratios, bullish flag | 7 |
| **Total** | | **66** |

---

## Quickstart

### Prerequisites

- Python 3.11+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Parthshewale18/BTC.git
cd bitcoin-ml-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Historical Data

```bash
python src/data_collection.py
```

Downloads ~173 days of BTC/USDT 5-minute candles from Binance.
Saves to `data/raw/btc_5m.csv`.

No API key required — Binance public endpoints are free.

### 4. Clean Data + Engineer Features

Open and run all cells in order:

```
notebooks/data_cleaning.ipynb          → saves data/processed/btc_5m_clean.csv
notebooks/feature_engineering.ipynb    → saves data/processed/btc_5m_features.csv
```

### 5. Train the Models

```
notebooks/train_model.ipynb            → saves models/*.pkl
```

This trains both XGBoost models and runs a backtest.
Saved model files will appear in `models/`.

### 6. Run a Single Prediction

```bash
python src/predict.py
```

Output:
```
====================================================
  Bitcoin ML  —  Real-Time Prediction System
====================================================
  Mode        : SINGLE RUN
  Symbol      : BTCUSDT
  Threshold   : 0.55

INFO | Fetched 200 candles | Latest close: $79,039.48
INFO | Computing technical indicators...
INFO | Models loaded | Features: 66

================================================
   BTC/USDT  --  Real-Time Prediction
================================================
  Time       : 2026-05-16T10:35:00 UTC
  Current    : $    79,039.48
  Predicted  : $    79,105.22  (+0.083%)
  Direction  : UP  (MODERATE)
  Confidence : [############--------] 61.2%
  Trustworthy: [OK]  (threshold: 55%)
================================================
```

### 7. Run Continuous Predictions (every 5 minutes)

```bash
python src/predict.py --loop
```

### 8. Start the API Server

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- **Swagger UI** → http://localhost:8000/docs
- **Live prediction** → http://localhost:8000/predict
- **Health check** → http://localhost:8000/health

---

## API Reference

### `GET /`
Welcome message and available endpoints.

```bash
curl http://localhost:8000/
```

### `GET /health`
Returns model loading status and API health.

```json
{
  "status": "ok",
  "api_version": "1.0.0",
  "models_loaded": true,
  "model_features": 66,
  "classifier_ready": true,
  "regressor_ready": true,
  "timestamp": "2026-05-16T10:35:00+00:00"
}
```

### `GET /predict`
Fetches live BTC data and returns a prediction.

```bash
curl http://localhost:8000/predict
```

```json
{
  "timestamp": "2026-05-16T10:35:00+00:00",
  "current_price": 79039.48,
  "predicted_price": 79105.22,
  "price_change_est": 65.74,
  "pct_change_est": 0.0832,
  "direction": "UP",
  "confidence": 0.6124,
  "signal_strength": "MODERATE",
  "trustworthy": true,
  "threshold_used": 0.55,
  "fetched_at": "2026-05-16T10:35:01+00:00"
}
```

**Signal strength guide:**

| Confidence | Signal Strength | Meaning |
|------------|----------------|---------|
| > 0.65 | STRONG | Model is very confident |
| 0.58–0.65 | MODERATE | Reasonable confidence |
| 0.50–0.58 | WEAK | Model is uncertain |
| < 0.55 | trustworthy: false | Do not act on this signal |

### `GET /history?limit=10`
Returns the last N predictions.

```bash
curl http://localhost:8000/history?limit=5
```

### `GET /history/stats`
Summary statistics across all saved predictions.

```json
{
  "total_predictions": 288,
  "up_predictions": 152,
  "down_predictions": 136,
  "up_pct": 52.78,
  "avg_confidence": 0.5312,
  "trustworthy_pct": 23.61,
  "first_prediction": "2026-05-15T00:00:00+00:00",
  "last_prediction": "2026-05-16T10:35:00+00:00"
}
```

---

## Docker

### Build and Run Locally

```bash
# Build the image
docker build -t bitcoin-ml .

# Run the container
docker run -p 8000:8000 bitcoin-ml

# Test it
curl http://localhost:8000/health

# View logs
docker logs $(docker ps -q --filter ancestor=bitcoin-ml)

# Stop
docker stop $(docker ps -q --filter ancestor=bitcoin-ml)
```

### What the Image Contains

```
Base        : python:3.11-slim
System libs : build-essential, libgomp1 (for XGBoost)
Python pkgs : all from requirements.txt
Code        : src/, app/
Models      : models/*.pkl (baked in at build time)
Port        : 8000
Health check: GET /health every 30s
```

---

## Deploy to Render

See [DEPLOY.md](DEPLOY.md) for the complete step-by-step guide.

**Quick summary:**

1. Push project to GitHub (including `models/` folder)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set **Runtime = Docker**
5. Click **Create Web Service**
6. Wait 3–5 minutes for build
7. Visit `https://your-app.onrender.com/docs`

---

## Best Practices Followed

### No Data Leakage
```python
# Target uses shift(-1) — looks forward, but only as the LABEL
df['next_close'] = df['close'].shift(-1)

# All features use shift(+N) — look backward only
df['close_lag_1'] = df['close'].shift(1)
```

### Time-Series Train/Test Split
```python
# NO shuffling — older data trains, newer data tests
split = int(len(df) * 0.80)
X_train = X.iloc[:split]   # older candles
X_test  = X.iloc[split:]   # newer candles
```

### Scaler Fitted on Train Only
```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # learn from train only
X_test_sc  = scaler.transform(X_test)       # apply same stats to test
```

### Confidence Threshold
```python
# Only trust predictions above 55% confidence
trustworthy = confidence >= 0.55
```

### Modular Code
```
data_collection.py  → only fetches data
predict.py          → only runs predictions
app.py              → only handles HTTP requests
notebooks/          → exploration and training
```

---

## Known Limitations

| Limitation | Details |
|------------|---------|
| No trading fees | Backtest does not subtract Binance fees (~0.1% per trade) |
| Long-only strategy | Backtest only goes long, never short |
| Single regime | Model trained on one market period may underperform in different regimes |
| No retraining | Models are static — market evolves, periodic retraining needed |
| 5-min lag | Prediction is for the NEXT 5-min candle close only |

---

## Roadmap / Future Improvements

- [ ] Add LSTM model as alternative regressor
- [ ] Walk-forward cross-validation (multiple train/test windows)
- [ ] Add short-selling to backtest strategy
- [ ] Automated retraining pipeline (weekly)
- [ ] Telegram / Discord alerts when high-confidence signal fires
- [ ] Add order book depth features
- [ ] Multi-coin support (ETH, SOL, BNB)
- [ ] Frontend dashboard (React + Chart.js)

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| Data | Binance REST API |
| Processing | pandas, numpy |
| Indicators | ta (Technical Analysis library) |
| ML Models | XGBoost |
| Scaling | scikit-learn StandardScaler |
| API | FastAPI + uvicorn |
| Container | Docker |
| Deployment | Render |
| Notebooks | Jupyter |

---

## Step-by-Step Learning Path

This project was built in 9 steps — each step builds on the previous:

| Step | File | What You Learn |
|------|------|---------------|
| 1 | Architecture | Problem framing, system design |
| 2 | data_collection.py | REST APIs, pagination, rate limiting |
| 3 | data_cleaning.ipynb | Data quality, target creation, leakage prevention |
| 4 | feature_engineering.ipynb | Technical indicators, rolling windows, lags |
| 5 | train_model.ipynb | XGBoost, time-series split, evaluation metrics, backtesting |
| 6 | predict.py | Real-time inference, logging, loop design |
| 7 | app/app.py | FastAPI, REST design, Pydantic schemas |
| 8 | Dockerfile | Containerisation, Docker layers, deployment |
| 9 | README.md | Documentation, project structure |

---

## Author

Built as a learning project for end-to-end ML engineering —
from raw API data to a deployed, containerised prediction API.

---

## License

MIT License — free to use, modify, and distribute.
