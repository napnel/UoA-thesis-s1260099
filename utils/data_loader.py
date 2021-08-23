import pandas as pd
import yfinance as yf
from typing import Optional


class DataLoader:
    @classmethod
    def load_data(self, filename: str) -> pd.DataFrame:
        data = pd.read_csv(filename, parse_dates=[0]).set_index("Date")
        return data

    @classmethod
    def split_data(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df.loc[start:end, :]

    @classmethod
    def fetch_data(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d"):
        """
        Feath historical data (OHLCV) using Yahoo Finance API.

        param ticker: ex) MSFT, ^N255, BTC-USD,
        param start: ex) Download start date string (YYYY-MM-DD) or _datetime. Default is 1900-01-01
        param end: ex) Download end date string (YYYY-MM-DD) or _datetime. Default is now
        param interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo Intraday data cannot
        """
        data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
        if interval != "1d":
            data = data.reset_index().rename(columns={"index": "Date"})
            data["Date"] = data["Date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))
            data = data.set_index("Date")

        assert data.index.name == "Date"
        assert list(data.columns.values) == ["Open", "High", "Low", "Close", "Volume"]
        return data
