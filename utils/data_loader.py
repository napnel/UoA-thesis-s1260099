import pandas as pd
import yfinance as yf

class DataLoader:
    @classmethod
    def load_data(self, filename: str) -> pd.DataFrame:
        data = pd.read_csv(filename, parse_dates=[0]).set_index("Date")
        return data

    @classmethod
    def split_data(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df.loc[start:end, :]

    @classmethod
    def fetch_data(self, ticker: str, start: str, end: str, interval: str = "1d", save=True) -> pd.DataFrame:
        """
        if interval is 1h: you can only fetch 730 days.
        if interval is 15m: you can only fetch 60 days.
        """
        data = yf.download(ticker, start=start, end=end, interval=interval)
        data = data.reset_index().rename(columns={"index": "Date"})
        if interval != "1d":
            data["Date"] = data["Date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))
        data["Close"] = data["Adj Close"]
        data = data.drop(["Adj Close"], axis=1)
        data = data.set_index("Date")
        return data
