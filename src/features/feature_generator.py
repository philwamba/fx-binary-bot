import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from typing import List

class FeatureGenerator:
    """
    Feature Engineering class for generating technical indicators
    and time-based features from OHLC data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLC DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with columns [Open, High, Low, Close, Volume]
        """
        self.df = df.copy()
        
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Add standard technical indicators using the 'ta' library.
        """
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        
        # RSI
        rsi = RSIIndicator(close=close, window=14)
        self.df['rsi'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        self.df['bb_high'] = bb.bollinger_hband()
        self.df['bb_low'] = bb.bollinger_lband()
        self.df['bb_mid'] = bb.bollinger_mavg()
        self.df['bb_width'] = (self.df['bb_high'] - self.df['bb_low']) / self.df['bb_mid']
        
        # MACD
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()
        
        # ATR (Average True Range)
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        self.df['atr'] = atr.average_true_range()
        
        # EMA
        ema_short = EMAIndicator(close=close, window=9)
        ema_long = EMAIndicator(close=close, window=21)
        self.df['ema_9'] = ema_short.ema_indicator()
        self.df['ema_21'] = ema_long.ema_indicator()
        
        return self.df
    
    def add_price_features(self) -> pd.DataFrame:
        """
        Add price-based features like returns and lagged values.
        """
        # Returns
        self.df['returns'] = self.df['Close'].pct_change()
        self.df['log_returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            self.df[f'returns_lag_{lag}'] = self.df['returns'].shift(lag)
        
        # Rolling statistics
        self.df['rolling_mean_5'] = self.df['Close'].rolling(window=5).mean()
        self.df['rolling_std_5'] = self.df['Close'].rolling(window=5).std()
        self.df['rolling_mean_20'] = self.df['Close'].rolling(window=20).mean()
        
        return self.df
    
    def add_time_features(self) -> pd.DataFrame:
        """
        Add time-based features (hour, day of week, etc.).
        Critical for FX markets which have strong time-of-day patterns.
        """
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex for time features")
        
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['is_monday'] = (self.df.index.dayofweek == 0).astype(int)
        self.df['is_friday'] = (self.df.index.dayofweek == 4).astype(int)
        
        # Encode cyclical features (hour)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        return self.df
    
    def generate_all_features(self) -> pd.DataFrame:
        """
        Generate all features: technical, price, and time-based.
        """
        self.add_technical_indicators()
        self.add_price_features()
        self.add_time_features()
        
        # Drop rows with NaN values (caused by indicators and lags)
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        print(f"Feature generation complete. Dropped {initial_rows - len(self.df)} rows with NaN.")
        
        return self.df
