"""
FastAPI Backend for FX Binary Options Bot

Endpoints:
- GET /api/status - Bot status
- POST /api/predict - Get prediction for current market
- GET /api/backtest/results - Latest backtest results
- GET /api/performance - Performance metrics
- GET /api/trades - Trade history
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader.yfinance_loader import YFinanceLoader
from features.feature_generator import FeatureGenerator
from strategy.signal_generator import SignalGenerator

app = FastAPI(title="FX Binary Options Bot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:57969"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = None
MODEL = None

@app.on_event("startup")
async def load_model():
    global MODEL_PATH, MODEL
    model_files = [f for f in os.listdir("models") if f.endswith('.joblib')]
    if model_files:
        MODEL_PATH = os.path.join("models", sorted(model_files)[-1])
        MODEL = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded: {MODEL_PATH}")
    else:
        print("⚠️  No model found. Run train.py first.")

# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionRequest(BaseModel):
    symbol: str = "EURUSD=X"
    
class PredictionResponse(BaseModel):
    signal: str
    probability: float
    prob_up: float
    prob_down: float
    timestamp: str
    current_price: float

class BacktestResult(BaseModel):
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_return: float
    final_balance: float
    break_even_rate: float

class Trade(BaseModel):
    timestamp: str
    action: str
    probability: float
    outcome: str
    profit: float
    balance: float

# ============================================================================
# Helper Functions
# ============================================================================

def get_latest_data(symbol: str = "EURUSD=X", hours: int = 100):
    """Fetch recent market data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    
    loader = YFinanceLoader()
    df = loader.fetch_history(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1h"
    )
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from raw data"""
    feature_gen = FeatureGenerator(df)
    df_features = feature_gen.generate_all_features()
    
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df_features.drop(columns=exclude_cols, errors='ignore')
    
    return X, df_features

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    return {"message": "FX Binary Options Bot API", "status": "running"}

@app.get("/api/status")
def get_status():
    """Get bot status"""
    return {
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate a trading signal for the current market"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get latest data
        df = get_latest_data(symbol=request.symbol)
        
        # Generate features
        X, df_features = prepare_features(df)
        
        # Get most recent data point
        latest_features = X.tail(1)
        current_price = df_features['Close'].iloc[-1]
        
        # Generate signal
        signal_gen = SignalGenerator(model=MODEL, probability_threshold=0.60)
        signal = signal_gen.generate_signal(latest_features)
        
        return PredictionResponse(
            signal=signal['action'],
            probability=float(signal['probability']),
            prob_up=float(signal['prob_up']),
            prob_down=float(signal['prob_down']),
            timestamp=datetime.now().isoformat(),
            current_price=float(current_price)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/results", response_model=BacktestResult)
def get_backtest_results():
    """Get latest backtest results"""
    result_files = [f for f in os.listdir("results") if f.endswith('.csv')]
    if not result_files:
        raise HTTPException(status_code=404, detail="No backtest results found")
    
    latest_result = os.path.join("results", sorted(result_files)[-1])
    df = pd.read_csv(latest_result)
    
    total_trades = len(df)
    wins = (df['outcome'] == 'WIN').sum()
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    initial_balance = 10000.0  # From config
    final_balance = df['balance'].iloc[-1] if len(df) > 0 else initial_balance
    net_return = (final_balance - initial_balance) / initial_balance
    
    return BacktestResult(
        total_trades=total_trades,
        wins=int(wins),
        losses=int(losses),
        win_rate=win_rate,
        net_return=net_return,
        final_balance=float(final_balance),
        break_even_rate=0.5405
    )

@app.get("/api/trades", response_model=List[Trade])
def get_trades(limit: int = 100):
    """Get trade history"""
    result_files = [f for f in os.listdir("results") if f.endswith('.csv')]
    if not result_files:
        raise HTTPException(status_code=404, detail="No trade history found")
    
    latest_result = os.path.join("results", sorted(result_files)[-1])
    df = pd.read_csv(latest_result)
    
    # Get last N trades
    df = df.tail(limit)
    
    trades = []
    for _, row in df.iterrows():
        trades.append(Trade(
            timestamp=row['timestamp'],
            action=row['action'],
            probability=float(row['probability']),
            outcome=row['outcome'],
            profit=float(row['profit']),
            balance=float(row['balance'])
        ))
    
    return trades

@app.get("/api/performance")
def get_performance():
    """Get performance metrics with chart data"""
    result_files = [f for f in os.listdir("results") if f.endswith('.csv')]
    if not result_files:
        raise HTTPException(status_code=404, detail="No results found")
    
    latest_result = os.path.join("results", sorted(result_files)[-1])
    df = pd.read_csv(latest_result)
    
    # Equity curve
    equity_curve = df[['timestamp', 'balance']].to_dict('records')
    
    # Win/loss distribution by hour
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    hourly_stats = df.groupby('hour').agg({
        'outcome': lambda x: (x == 'WIN').sum() / len(x) if len(x) > 0 else 0
    }).reset_index()
    hourly_stats.columns = ['hour', 'win_rate']
    
    # Action distribution
    action_dist = df['action'].value_counts().to_dict()
    
    return {
        "equity_curve": equity_curve,
        "hourly_win_rate": hourly_stats.to_dict('records'),
        "action_distribution": action_dist,
        "total_trades": len(df),
        "avg_profit_per_trade": float(df['profit'].mean())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
