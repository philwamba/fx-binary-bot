"""
Training Script for FX Binary Options Bot

This script orchestrates the full training pipeline:
1. Load historical data
2. Generate features
3. Create labels
4. Perform walk-forward validation
5. Train final model
"""

import os
import sys
from datetime import datetime
from xgboost import XGBClassifier

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader.yfinance_loader import YFinanceLoader
from features.feature_generator import FeatureGenerator
from features.labeler import BinaryOptionsLabeler
from models.walk_forward_validator import WalkForwardValidator

def main():
    print("\n" + "="*80)
    print("FX BINARY OPTIONS BOT - TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    SYMBOL = "EURUSD=X"  # Yahoo Finance symbol for EUR/USD
    START_DATE = "2024-01-01"
    END_DATE = "2024-12-01"
    INTERVAL = "1h"  # 1-hour candles
    EXPIRATION_PERIODS = 5  # 5-period binary options
    
    # Model parameters
    MODEL_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    PROBABILITY_THRESHOLD = 0.60  # Only trade if probability > 60%
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    print("STEP 1: Loading historical data...")
    loader = YFinanceLoader()
    df = loader.fetch_history(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL
    )
    
    print(f"Loaded {len(df):,} rows of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Shape: {df.shape}\n")
    
    # ============================================================================
    # STEP 2: GENERATE FEATURES
    # ============================================================================
    print("STEP 2: Generating features...")
    feature_gen = FeatureGenerator(df)
    df_features = feature_gen.generate_all_features()
    
    print(f"Features generated: {df_features.shape[1]} columns")
    print(f"Rows after cleaning: {len(df_features):,}\n")
    
    # ============================================================================
    # STEP 3: CREATE LABELS
    # ============================================================================
    print("STEP 3: Creating labels...")
    labeler = BinaryOptionsLabeler(df_features, expiration_periods=EXPIRATION_PERIODS)
    df_labeled = labeler.create_labels()
    
    X, y = labeler.get_features_and_target()
    
    print(f"\nDataset ready:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y):   {y.shape}")
    print(f"  Feature names: {list(X.columns)[:10]}...\n")
    
    # ============================================================================
    # STEP 4: WALK-FORWARD VALIDATION
    # ============================================================================
    print("STEP 4: Walk-Forward Validation...")
    validator = WalkForwardValidator(n_splits=5, gap=0)
    
    results = validator.validate(
        X=X,
        y=y,
        model_class=XGBClassifier,
        model_params=MODEL_PARAMS
    )
    
    # ============================================================================
    # STEP 5: TRAIN FINAL MODEL
    # ============================================================================
    print("STEP 5: Training final model on all data...")
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/xgb_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    
    final_model = validator.train_final_model(
        X=X,
        y=y,
        model_class=XGBClassifier,
        model_params=MODEL_PARAMS,
        save_path=model_path
    )
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved to: {model_path}")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
    print(f"Mean F1 Score: {results['mean_f1']:.4f}")
    print("\nNext steps:")
    print("  1. Review the validation results above")
    print("  2. Adjust model parameters if needed")
    print("  3. Run backtesting with main.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
