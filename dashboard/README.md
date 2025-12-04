# Trading Bot Dashboard

Beautiful real-time dashboard for monitoring the FX Binary Options Bot.

## Features

- 🎯 Live market predictions with confidence scores
- 📊 Interactive charts (Equity curve, hourly performance, action distribution)
- 📈 Real-time performance metrics
- 📋 Recent trade history
- 🎨 Modern glassmorphism design
- 🔄 Auto-refresh every 60 seconds

## Setup

### 1. Install Python Dependencies

```bash
cd ..
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python api.py
```

The API will start on `http://localhost:8000`

### 3. Open the Dashboard

Simply open `index.html` in your browser:

```bash
open index.html  # macOS
# or
start index.html # Windows
# or
xdg-open index.html # Linux
```

## API Endpoints

- `GET /api/status` - Check bot status
- `POST /api/predict` - Get live trading signal
- `GET /api/backtest/results` - Latest backtest metrics
- `GET /api/performance` - Performance data with charts
- `GET /api/trades?limit=20` - Recent trades

##Screenshots

The dashboard features:

1. **Live Prediction Card** - Real-time market signals with confidence
2. **Performance Metrics** - Win rate, total trades, returns, balance
3. **Equity Curve** - Visual representation of balance over time
4. **Hourly Analysis** - Win rate breakdown by hour of day
5. **Action Distribution** - CALL vs PUT trade distribution
6. **Trade History** - Recent trades with outcomes and P&L

## Customization

Edit the `index.html` to customize:

- Colors and gradients
- Chart types and styles
- Refresh intervals
- Number of trades displayed
