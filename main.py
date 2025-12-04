"""
Backtest Script for FX Binary Options Bot

This script runs a backtest simulation:
1. Load historical data
2. Load trained model
3. Generate signals
4. Simulate paper trading with PROPER evaluation
5. Report results
"""

import os
import sys
from datetime import datetime
import joblib
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader.yfinance_loader import YFinanceLoader
from features.feature_generator import FeatureGenerator
from strategy.signal_generator import SignalGenerator
from execution.paper_trader import PaperTrader

def main():
    print("\n" + "="*80)
    print("FX BINARY OPTIONS BOT - BACKTEST")
    print("="*80 + "\n")
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    SYMBOL = "EURUSD=X"
    START_DATE = "2024-01-01"  # Use more data for robust backtest
    END_DATE = "2024-12-01"
    INTERVAL = "1h"
    EXPIRATION_PERIODS = 5
    
    # Find latest model
    model_files = [f for f in os.listdir("models") if f.endswith('.joblib')]
    if not model_files:
        print("ERROR: No trained model found in models/ directory")
        print("Please run train.py first.")
        return
    
    model_path = os.path.join("models", sorted(model_files)[-1])
    print(f"Loading model: {model_path}\n")
    
    model = joblib.load(model_path)
    
    # Paper trading settings
    INITIAL_BALANCE = 10000.0
    PAYOUT_RATIO = 0.85  # 85% payout on winning trades
    TRADE_AMOUNT = 10.0
    PROBABILITY_THRESHOLD = 0.60
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    print("STEP 1: Loading backtest data...")
    loader = YFinanceLoader()
    df = loader.fetch_history(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL
    )
    
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}\n")
    
    # ============================================================================
    # STEP 2: GENERATE FEATURES
    # ============================================================================
    print("STEP 2: Generating features...")
    feature_gen = FeatureGenerator(df)
    df_features = feature_gen.generate_all_features()
    
    # ============================================================================
    # STEP 3: EXTRACT FEATURES (NO LABELS NEEDED FOR BACKTEST)
    # ============================================================================
    print("STEP 3: Preparing features for prediction...")
    
    # Extract features only (no labels)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df_features.drop(columns=exclude_cols, errors='ignore')
    
    # Keep the original close prices for evaluation
    close_prices = df_features['Close'].copy()
    
    print(f"Features: {X.shape}")
    print(f"Close prices: {len(close_prices)}\\n")
    
    # ============================================================================
    # STEP 4: WALK-FORWARD BACKTEST (PROPER EVALUATION)
    # ============================================================================
    print("STEP 4: Running walk-forward backtest...")
    print("Note: Only trading when we have future data to evaluate\\n")
    
    signal_gen = SignalGenerator(model=model, probability_threshold=PROBABILITY_THRESHOLD)
    paper_trader = PaperTrader(
        initial_balance=INITIAL_BALANCE,
        payout_ratio=PAYOUT_RATIO,
        trade_amount=TRADE_AMOUNT
    )
    
    # We can only trade up to (len - EXPIRATION_PERIODS) because we need future data to evaluate
    max_trade_idx = len(X) - EXPIRATION_PERIODS
    
    trades_executed = 0
    for i in range(max_trade_idx):
        # Get features for current time step
        current_features = X.iloc[i:i+1]
        
        # Generate signal
        signal = signal_gen.generate_signal(current_features)
        
        if signal['action'] == 'NO_TRADE':
            continue
        
        # Get current and future price
        current_price = close_prices.iloc[i]
        future_price = close_prices.iloc[i + EXPIRATION_PERIODS]
        
        # Determine actual outcome based on ACTUAL price movement
        # NOT based on pre-computed labels!
        if signal['action'] == 'CALL':
            actual_outcome = 1 if future_price > current_price else 0
        elif signal['action'] == 'PUT':
            actual_outcome = 1 if future_price < current_price else 0
        else:
            continue
        
        # Execute trade with actual outcome
        paper_trader.execute_trade(
            signal=signal,
            actual_outcome=actual_outcome,
            timestamp=df_features.index[i]
        )
        trades_executed += 1
    
    print(f"Total possible trades: {max_trade_idx:,}")
    print(f"Trades executed: {trades_executed:,}\\n")
    
    # ============================================================================
    # STEP 5: RESULTS
    # ============================================================================
    paper_trader.print_summary()
    
    # Save trade history
    os.makedirs("results", exist_ok=True)
    trade_log_path = f"results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    paper_trader.save_trades(trade_log_path)
    
    # ============================================================================
    # ANALYSIS
    # ============================================================================
    stats = paper_trader.get_statistics()
    
    # Break-even win rate calculation
    breakeven_rate = 1 / (1 + PAYOUT_RATIO)
    
    print(f"Break-even win rate: {breakeven_rate:.2%}")
    print(f"Actual win rate:     {stats['win_rate']:.2%}")
    
    if stats['win_rate'] > breakeven_rate:
        print("✓ Strategy is PROFITABLE (above break-even)")
    else:
        print("✗ Strategy is NOT PROFITABLE (below break-even)")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
