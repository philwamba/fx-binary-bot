from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Tuple

class AbstractDataProvider(ABC):
    """
    Abstract base class for data providers.
    Enforces a standard interface for fetching historical and real-time data.
    """

    @abstractmethod
    def fetch_history(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Fetch historical OHLC data.
        
        Args:
            symbol (str): The asset symbol (e.g., "EURUSD=X").
            start_date (str): Start date in "YYYY-MM-DD" format.
            end_date (str): End date in "YYYY-MM-DD" format.
            interval (str): Data interval (e.g., "1m", "5m", "1h").
            
        Returns:
            pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex.
        """
        pass

    @abstractmethod
    def stream_data(self, symbol: str):
        """
        Stream real-time data (generator or callback based).
        """
        pass
