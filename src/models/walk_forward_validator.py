import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Tuple, Dict
import joblib
from datetime import datetime

class WalkForwardValidator:
    """
    Walk-Forward Validation for time-series models.
    Prevents lookahead bias by training on past data and testing on future unseen data.
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Initialize validator.
        
        Args:
            n_splits (int): Number of train/test splits
            gap (int): Gap between train and test sets (to simulate real-world delay)
        """
        self.n_splits = n_splits
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        self.results = []
    
    def validate(self, X: pd.DataFrame, y: pd.Series, model_class, model_params: dict = None) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target labels
            model_class: Scikit-learn compatible model class (e.g., XGBClassifier)
            model_params (dict): Model hyperparameters
        
        Returns:
            Dict: Validation results
        """
        if model_params is None:
            model_params = {}
        
        fold_results = []
        
        print(f"\n{'='*60}")
        print(f"Starting Walk-Forward Validation with {self.n_splits} splits")
        print(f"{'='*60}\n")
        
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            metrics = {
                'fold': fold,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            fold_results.append(metrics)
            
            print(f"Fold {fold}/{self.n_splits}:")
            print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
            print(f"  Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print()
        
        # Aggregate results
        df_results = pd.DataFrame(fold_results)
        
        summary = {
            'mean_accuracy': df_results['accuracy'].mean(),
            'std_accuracy': df_results['accuracy'].std(),
            'mean_precision': df_results['precision'].mean(),
            'mean_recall': df_results['recall'].mean(),
            'mean_f1': df_results['f1'].mean(),
            'fold_details': fold_results
        }
        
        if 'roc_auc' in df_results.columns:
            summary['mean_roc_auc'] = df_results['roc_auc'].mean()
        
        print(f"{'='*60}")
        print(f"SUMMARY - Walk-Forward Validation Results")
        print(f"{'='*60}")
        print(f"Mean Accuracy:  {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        print(f"Mean Precision: {summary['mean_precision']:.4f}")
        print(f"Mean Recall:    {summary['mean_recall']:.4f}")
        print(f"Mean F1:        {summary['mean_f1']:.4f}")
        if 'mean_roc_auc' in summary:
            print(f"Mean ROC-AUC:   {summary['mean_roc_auc']:.4f}")
        print(f"{'='*60}\n")
        
        self.results = fold_results
        return summary
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, model_class, model_params: dict = None, save_path: str = None):
        """
        Train final model on all available data.
        
        Args:
            X (pd.DataFrame): Full feature matrix
            y (pd.Series): Full target
            model_class: Model class
            model_params (dict): Model parameters
            save_path (str): Path to save the model
        
        Returns:
            Trained model
        """
        if model_params is None:
            model_params = {}
        
        print(f"\nTraining final model on {len(X):,} samples...")
        model = model_class(**model_params)
        model.fit(X, y)
        
        if save_path:
            joblib.dump(model, save_path)
            print(f"Model saved to: {save_path}")
        
        return model
