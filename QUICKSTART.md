# Quick Start Guide

## 🚀 Run the Complete System in 3 Steps

### Step 1: Train the Model

```bash
python train.py
```

✅ Downloads EUR/USD data  
✅ Generates 26 features  
✅ Trains XGBoost model  
**Time**: ~1 minute

### Step 2: Start the API

```bash
python api.py
```

✅ Loads trained model  
✅ Starts on `http://localhost:8000`  
**Keep this terminal open**

### Step 3: Open Dashboard

```bash
open dashboard/index.html
```

✅ Beautiful web UI  
✅ Real-time predictions  
✅ Interactive charts

---

## 📸 Dashboard Preview

![Dashboard Screenshot](file:///Users/filwillian/.gemini/antigravity/brain/44d58a02-315b-49fa-8089-2be799e77dc9/dashboard_mockup_1764873846771.png)

**Features:**

- 🎯 Live trading signals with confidence
- 📊 Equity curve visualization
- 📈 Hourly performance analysis
- 📋 Recent trade history
- 🎨 Glassmorphism design

---

## 🔧 What Was Fixed

### Critical Bug: Data Leakage

**Before**: 98.72% win rate (impossible)  
**After**: 52.26% win rate (realistic)

**The Issue**: Backtest was using future data to evaluate trades.

**The Fix**: [main.py](file:///Users/filwillian/Projects/Python/fx-binary-bot/main.py) now correctly evaluates based on actual price movement:

```python
# Correct evaluation
current_price = close_prices.iloc[i]
future_price = close_prices.iloc[i + EXPIRATION_PERIODS]
actual_outcome = 1 if future_price > current_price else 0
```

---

## 📊 Real Performance (After Fix)

| Metric       | Value   |
| ------------ | ------- |
| Win Rate     | 52.26%  |
| Break-even   | 54.05%  |
| Total Trades | 3,961   |
| Net Return   | -13.15% |

**Status**: Below break-even, but architecture is solid for improvement.

---

## 🎯 API Endpoints

```bash
# Check status
curl http://localhost:8000/api/status

# Get prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "EURUSD=X"}'

# View backtest results
curl http://localhost:8000/api/backtest/results

# Get trade history
curl http://localhost:8000/api/trades?limit=20
```

---

## 💡 Next Steps to Improve

1. **Add More Features** (volume, sentiment, economic calendar)
2. **Hyperparameter Tuning** (Optuna)
3. **Ensemble Models** (XGBoost + LightGBM)
4. **Risk Management** (position sizing, stop loss)
5. **Live Data** (replace YFinance with real-time WebSocket)

---

## 📁 Project Structure

```
fx-binary-bot/
├── api.py              # FastAPI backend
├── train.py            # ML training pipeline
├── main.py             # Backtest (FIXED)
├── dashboard/
│   └── index.html      # Web UI
└── src/                # Modular components
```

---

## 🐛 Troubleshooting

**Dashboard shows "API Offline"**

```bash
# Check API is running
python api.py
```

**No predictions**

```bash
# Train model first
python train.py
```

**Port 8000 already in use**

```bash
lsof -i :8000
kill -9 <PID>
```

---

**Ready to use!** 🎉
