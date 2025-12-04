# FX Binary Options Trading Bot

An advanced, modular trading research bot for currency binary options that ingests real-time FX data, computes technical indicators, trains machine learning models with walk-forward validation, and generates probability-filtered trading signals.

## Features

- **Modular Architecture**: Clean separation between data loading, feature engineering, modeling, and execution
- **Walk-Forward Validation**: Robust time-series validation to prevent overfitting
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, EMAs, and custom features
- **Machine Learning**: XGBoost classifier with probability-based signal filtering
- **Paper Trading**: Built-in simulation to test strategies without risk
- **Time-Based Features**: Hour-of-day and day-of-week encoding for FX market patterns

## Project Structure

```
fx-binary-bot/
├── data/                   # Data storage (ignored in git)
├── src/
│   ├── data_loader/        # Data ingestion (YFinance, future: WebSocket)
│   ├── features/           # Feature engineering and labeling
│   ├── models/             # ML training and walk-forward validation
│   ├── strategy/           # Signal generation
│   ├── execution/          # Paper trading
│   └── utils/              # Utilities
├── notebooks/              # Jupyter notebooks for research
├── tests/                  # Unit tests
├── models/                 # Saved trained models
├── results/                # Backtest results and logs
├── train.py                # Training script
├── main.py                 # Backtest script
└── requirements.txt        # Dependencies
```

## Installation

1. Clone the repository:

```bash
git clone git@github.com:philwamba/fx-binary-bot.git
cd fx-binary-bot
```

2. Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train a Model

Run the training pipeline to download data, engineer features, and train a model:

```bash
python train.py
```

This will:

- Download EUR/USD data from Yahoo Finance
- Generate technical indicators and features
- Create binary labels based on 5-period price movement
- Perform walk-forward validation (5 splits)
- Train a final model and save it to `models/`

### 2. Run Backtest

Test the trained model on recent data:

```bash
python main.py
```

This will:

- Load the latest trained model
- Generate signals on backtest data
- Simulate paper trading
- Display performance metrics and save results to `results/`

### 3. Customize Configuration

Open `train.py` or `main.py` and modify:

- `SYMBOL`: Currency pair (e.g., "GBPUSD=X")
- `INTERVAL`: Data frequency ("1h", "5m", etc.)
- `EXPIRATION_PERIODS`: Binary option expiration (e.g., 5)
- `PROBABILITY_THRESHOLD`: Minimum confidence to trade (e.g., 0.60)
- `MODEL_PARAMS`: XGBoost hyperparameters

## Key Concepts

### Binary Options Labeling

```python
Target = 1 if Close(t + expiration_periods) > Close(t) else 0
```

For a 5-period binary option on 1-hour data, if the price after 5 hours is higher than the current price, the label is 1 (CALL), otherwise 0 (PUT).

### Walk-Forward Validation

Instead of random train/test split (which causes lookahead bias), we use time-series splits:

- **Fold 1**: Train on Jan-Mar, Test on Apr
- **Fold 2**: Train on Jan-Apr, Test on May
- **Fold 3**: Train on Jan-May, Test on Jun

This mimics real-world trading where you only know the past.

### Probability Filtering

The model outputs probabilities. We only trade if:

```
P(UP) > 0.60 → CALL
P(DOWN) > 0.60 → PUT
Otherwise → NO_TRADE
```

This filters out low-confidence signals.

### Break-Even Analysis

For an 85% payout ratio, the break-even win rate is:

```
Break-even = 1 / (1 + payout_ratio) = 1 / 1.85 = 54.05%
```

You need to win >54% of trades to be profitable.

## Performance Metrics

The bot tracks:

- **Accuracy**: % of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Win Rate**: % of profitable trades
- **Net Return**: Total profit / initial balance

## Next Steps

1. **Add More Data Sources**: Implement live WebSocket streaming
2. **Feature Engineering**: Add volume-based indicators, order flow data
3. **Hyperparameter Tuning**: Use Optuna or GridSearch for optimization
4. **Risk Management**: Add position sizing, Kelly Criterion
5. **Live Deployment**: Connect to a real binary options broker API

## Warning

⚠️ **This is a research tool for educational purposes only.** Trading binary options involves significant risk. Past performance does not guarantee future results. Always test thoroughly on paper before risking real capital.

## License

MIT License
