from typing import List
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


class FeatureEngineer:
    def __init__(self, technical_indicator_list: List[str]):
        self.technical_indicator_list = technical_indicator_list

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        # 欠損値処理
        data = self.clean_data(df)

        # To do: 外れ値の除去

        # 特徴抽出
        features = self.extract_features(data)

        return features

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        assert not data.isnull().sum().any()
        return data

    def extract_features(self, data: pd.DataFrame):
        features = pd.DataFrame()
        features["Price Percent Change"] = data["Close"].pct_change()
        features["Volume Percent Change"] = data["Volume"].pct_change()

        # Trend Indicators
        sma20 = SMAIndicator(data["Close"], window=20).sma_indicator()
        sma50 = SMAIndicator(data["Close"], window=50).sma_indicator()
        macd = MACD(data["Close"], window_slow=26, window_fast=12, window_sign=9)
        features["Diff Price SMA20 Percent Change"] = (data["Close"] - sma20) / sma20
        features["Diff Price SMA50 Percent Change"] = (data["Close"] - sma50) / sma50
        features["MACD Percent Change"] = (macd.macd() - macd.macd_signal()) / macd.macd_signal()

        # Oscillator Indicator
        rsi = RSIIndicator(data["Close"], window=14).rsi()
        features["RSI"] = rsi / 100.0

        # Volatility Indicator
        bb_sigma_1 = BollingerBands(data["Close"], window=20, window_dev=1)
        bb_sigma_2 = BollingerBands(data["Close"], window=20, window_dev=2)
        features["BB Sigma-1 Upper Bound"] = (data["Close"] - bb_sigma_1.bollinger_hband()) / bb_sigma_1.bollinger_hband()
        features["BB Sigma-1 Lower Bound"] = (data["Close"] - bb_sigma_1.bollinger_lband()) / bb_sigma_1.bollinger_lband()
        features["BB Sigma-2 Upper Bound"] = (data["Close"] - bb_sigma_2.bollinger_hband()) / bb_sigma_2.bollinger_hband()
        features["BB Sigma-2 Lower Bound"] = (data["Close"] - bb_sigma_2.bollinger_lband()) / bb_sigma_2.bollinger_lband()

        # Nanを0で埋める
        features = features.fillna(0)
        # features = df.fillna(method="bfill").fillna(method="ffill")
        return features


def main():
    import matplotlib.pyplot as plt

    df = pd.read_csv("./data/3600/ethusd/2021-01-01.csv", parse_dates=[0]).set_index("Date")

    feature_enginner = FeatureEngineer([])

    tmp = feature_enginner.preprocessing(df)
    print(tmp.head(50))
    tmp.hist()
    plt.show()


if __name__ == "__main__":
    main()
