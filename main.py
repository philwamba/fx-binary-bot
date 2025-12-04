"""
Backtest Script for FX Binary Options Bot

This script runs a backtest simulation:
1. Load historical data
2. Load trained model
3. Generate signals
4. Simulate paper trading
5. Report results
"""

import os
import sys
from datetime import datetime
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader.yfinance_loader import YFinanceLoader
from features.feature_generator import FeatureGenerator
from features.labeler import BinaryOptionsLabeler
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
    START_DATE = "2024-11-01"  # Use recent data for backtest
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
    # STEP 3: CREATE LABELS (for evaluation)
    # ============================================================================
    print("STEP 3: Creating labels for evaluation...")
    labeler = BinaryOptionsLabeler(df_features, expiration_periods=EXPIRATION_PERIODS)
    df_labeled = labeler.create_labels()
    
    X, y = labeler.get_features_and_target()
    
    print(f"Backtest dataset: {X.shape}\n")
    
    # ============================================================================
    # STEP 4: GENERATE SIGNALS & SIMULATE TRADING
    # ============================================================================
    print("STEP 4: Generating signals and simulating trades...")
    
    signal_gen = SignalGenerator(model=model, probability_threshold=PROBABILITY_THRESHOLD)
    paper_trader = PaperTrader(
        initial_balance=INITIAL_BALANCE,
        payout_ratio=PAYOUT_RATIO,
        trade_amount=TRADE_AMOUNT
    )
    
    # Generate signals for all data points
    signals = signal_gen.generate_signal(X)
    
    # Convert to list if single signal
    if isinstance(signals, dict):
        signals = [signals]
    
    # Execute trades
    trades_executed = 0
    for i, signal in enumerate(signals):
        if signal['action'] != 'NO_TRADE':
            paper_trader.execute_trade(
                signal=signal,
                actual_outcome=y.iloc[i],
                timestamp=df_labeled.index[i]
            )
            trades_executed += 1
    
    print(f"Signals generated: {len(signals)}")
    print(f"Trades executed: {trades_executed}\n")
    
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
    # For 85% payout: need to win (1/(1+0.85)) = 54.05% to break even
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
