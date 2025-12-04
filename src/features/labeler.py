import pandas as pd
import numpy as np

class BinaryOptionsLabeler:
    """
    Label generator for binary options.
    Creates target variable based on future price movement.
    """
    
    def __init__(self, df: pd.DataFrame, expiration_periods: int = 5):
        """
        Initialize labeler.
        
        Args:
            df (pd.DataFrame): Feature DataFrame with Close prices
            expiration_periods (int): Number of periods until expiration (e.g., 5 for 5-minute options)
        """
        self.df = df.copy()
        self.expiration_periods = expiration_periods
    
    def create_labels(self) -> pd.DataFrame:
        """
        Create binary labels: 1 if price goes UP, 0 if DOWN.
        
        Label logic:
            Target = 1 if Close(t + expiration_periods) > Close(t)
            Target = 0 otherwise
        """
        # Ensure Close is a Series (handle if yfinance returns DataFrame format)
        if isinstance(self.df['Close'], pd.DataFrame):
            close_series = self.df['Close'].iloc[:, 0]
        else:
            close_series = self.df['Close']
        
        # Shift close prices backwards to get future price
        future_close = close_series.shift(-self.expiration_periods)
        
        # Create binary target
        target = (future_close > close_series).astype(int)
        
        # Assign to dataframe
        self.df['future_close'] = future_close
        self.df['target'] = target
        
        # Drop rows where we don't have future data
        self.df = self.df.dropna(subset=['future_close'])
        
        # Calculate class distribution
        target_dist = self.df['target'].value_counts(normalize=True)
        print(f"\nTarget Distribution:")
        print(f"  UP (1):   {target_dist.get(1, 0):.2%}")
        print(f"  DOWN (0): {target_dist.get(0, 0):.2%}")
        
        return self.df
    
    def get_features_and_target(self, exclude_cols: list = None):
        """
        Split DataFrame into features (X) and target (y).
        
        Args:
            exclude_cols (list): Columns to exclude from features (e.g., ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        Returns:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target labels
        """
        if exclude_cols is None:
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close', 'target']
        else:
            exclude_cols = exclude_cols + ['future_close', 'target']
        
        X = self.df.drop(columns=exclude_cols, errors='ignore')
        y = self.df['target']
        
        return X, y
