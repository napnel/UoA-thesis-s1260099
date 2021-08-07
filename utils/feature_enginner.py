from cgi import test
from ctypes import Union
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureEngineer:
    def __init__(self, data_train: pd.DataFrame, data_val: pd.DataFrame, data_test: Optional[pd.DataFrame] = None, norm_method: str = "standard"):
        self.entire_raw: Dict[Optional[pd.DataFrame]] = {
            "train": data_train.copy(),
            "val": data_val.copy() if data_val else None,
            "test": data_test.copy() if data_test else None,
        }
        self.entire_data: Dict[Optional[pd.DataFrame]] = {"train": None, "val": None, "test": None}
        self.entire_features: Dict[Optional[pd.DataFrame]] = {"train": None, "val": None, "test": None}
        self.entire_normalized_features: Dict[Optional[pd.DataFrame]] = {"train": None, "val": None, "test": None}

        self.norm_method = norm_method
        if self.norm_method == "standard":
            self.scaler = StandardScaler()
        elif self.norm_method == "min-max":
            self.scaler = MinMaxScaler()
        elif self.norm_method == "robust":
            self.scaler = RobustScaler()
        assert self.scaler, f"Wrong Normalization Method, you use {self.norm_method}"

        self._run()

    def _run(self):
        for key, raw in self.entire_raw.items():
            if raw is not None:
                self.entire_data[key] = self.clean_data(raw)

        for key, data in self.entire_data.items():
            if data is not None:
                features = self.extract_features(data)
                self.entire_features[key] = features
                # Some value contain NaN when calculating the indicator.
                # Since NaN is dropped, align the start postion of data with feature
                self.entire_data[key] = data.loc[features.index[0] :, :]
                assert len(self.entire_data[key]) == len(features), f"data length: {len(self.entire_data[key])}, feature length: {len(features)}"

        self.scaler.fit(self.entire_features["train"])
        for key, features in self.entire_features.items():
            if features is not None:
                self.entire_normalized_features[key] = self.normalizing(features)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        assert not data.isnull().sum().any()
        return data

    def extract_features(self, data: pd.DataFrame):
        features = pd.DataFrame()
        features["Price Log Diff"] = np.log(data["Close"].shift(1)) - np.log(data["Close"])
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
        features["RSI"] = rsi / 100.0

        # Volatility Indicator
        bb_sigma_1 = BollingerBands(data["Close"], window=20, window_dev=1)
        bb_sigma_2 = BollingerBands(data["Close"], window=20, window_dev=2)
        features["BB Sigma-1 Upper Bound"] = np.log(data["Close"]) - np.log(bb_sigma_1.bollinger_hband())
        features["BB Sigma-1 Lower Bound"] = np.log(data["Close"]) - np.log(bb_sigma_1.bollinger_lband())
        features["BB Sigma-2 Upper Bound"] = np.log(data["Close"]) - np.log(bb_sigma_2.bollinger_hband())
        features["BB Sigma-2 Lower Bound"] = np.log(data["Close"]) - np.log(bb_sigma_2.bollinger_lband())

        return features.dropna()

    def normalizing(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(data), index=data.index, columns=data.columns)


def main():
    import matplotlib.pyplot as plt

    df = pd.read_csv("./data/3600/ethusd/2021-01-01.csv", parse_dates=[0]).set_index("Date")

    df_train = df.iloc[: len(df) // 2, :]
    df_val = df.iloc[len(df) // 2 :, :]

    print(df_train.shape, df_val.shape)
    feature_enginner = FeatureEngineer(df_train, df_val)

    normalized_features_train = feature_enginner.entire_normalized_features["train"]
    normalized_features_train.hist(figsize=(12, 10))
    plt.show()


if __name__ == "__main__":
    main()
