import yfinance as yf
import pandas as pd
from .base import AbstractDataProvider

class YFinanceLoader(AbstractDataProvider):
    """
    Data provider implementation using yfinance.
    Useful for research and backtesting with free data.
    """

    def fetch_history(self, symbol: str, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        """
        print(f"Fetching {symbol} from {start_date} to {end_date} at {interval} interval...")
        
        # yfinance expects interval as '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        df = yf.download(tickers=symbol, start=start_date, end=end_date, interval=interval, progress=False)
        
        if df.empty:
            print(f"Warning: No data found for {symbol}")
            return df

        # Ensure standard columns and index
        # yfinance returns: Open, High, Low, Close, Adj Close, Volume
        # We drop Adj Close for FX usually, or keep it. Let's stick to OHLCV.
        
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Ensure index is datetime and sorted
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Handle missing values if any (forward fill then backward fill)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        return df

    def stream_data(self, symbol: str):
        raise NotImplementedError("YFinance does not support true real-time streaming via this loader. Use a WebSocket provider.")
