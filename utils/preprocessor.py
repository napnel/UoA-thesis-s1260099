from cgi import test
from ctypes import Union
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Todo
# このクラスは特徴量だけを抽出するクラスにする
# 交差検証のことを考えると、ここでデータ分割してしまうと拡張性が失われる


class Preprocessor:
    @classmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        assert not data.isnull().sum().any()
        return data

    @classmethod
    def extract_features(self, data: pd.DataFrame):
        features = pd.DataFrame()
        features["Open Log Diff"] = np.log(data["Close"]) - np.log(data["Open"])
        features["High Log Diff"] = np.log(data["Close"]) - np.log(data["High"])
        features["Low Log Diff"] = np.log(data["Close"]) - np.log(data["Low"])
        features["Close Log Diff"] = np.log(data["Close"].shift(1)) - np.log(data["Close"])
        features["Volume Log Diff"] = np.log(data["Volume"].shift(1)) - np.log(data["Volume"])

        # Trend Indicators
        sma20 = SMAIndicator(data["Close"], window=20).sma_indicator()
        sma50 = SMAIndicator(data["Close"], window=50).sma_indicator()
        macd = MACD(data["Close"], window_slow=26, window_fast=12, window_sign=9)
        features["SMA-20 Log Diff"] = np.log(data["Close"]) - np.log(sma20)
        features["SMA-50 Log Diff"] = np.log(data["Close"]) - np.log(sma50)
        features["MACD"] = macd.macd_diff()

        # Oscillator Indicator
        rsi = RSIIndicator(data["Close"], window=14).rsi()
        features["RSI"] = rsi

        # Volatility Indicator
        bb_sigma_1 = BollingerBands(data["Close"], window=20, window_dev=1)
        bb_sigma_2 = BollingerBands(data["Close"], window=20, window_dev=2)
        features["BB Sigma-1 Upper Bound"] = np.log(data["Close"]) - np.log(bb_sigma_1.bollinger_hband())
        features["BB Sigma-1 Lower Bound"] = np.log(data["Close"]) - np.log(bb_sigma_1.bollinger_lband())
        features["BB Sigma-2 Upper Bound"] = np.log(data["Close"]) - np.log(bb_sigma_2.bollinger_hband())
        features["BB Sigma-2 Lower Bound"] = np.log(data["Close"]) - np.log(bb_sigma_2.bollinger_lband())

        return features

    @classmethod
    def align_date(self, data: pd.DataFrame, features: pd.DataFrame):
        features = features.dropna()
        data = data.loc[features.index[0] :, :]
        assert len(data) == len(features)
        return data, features


def main():
    import matplotlib.pyplot as plt

    df = pd.read_csv("./data/3600/ethusd/2021-01-01.csv", parse_dates=[0]).set_index("Date")

    df = Preprocessor.clean_data(df)
    features = Preprocessor.extract_features(df)
    df, features = Preprocessor.align_date(df, features)

    features.hist(figsize=(12, 10), bins=30)
    plt.show()


if __name__ == "__main__":
    main()
