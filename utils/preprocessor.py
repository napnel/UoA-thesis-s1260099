from pyexpat import features
import warnings
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

from ta.trend import sma_indicator, MACD, ADXIndicator, STCIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import money_flow_index, chaikin_money_flow


class Preprocessor:
    @classmethod
    def preprocessing(self, _data_train: pd.DataFrame, _data_eval: pd.DataFrame):
        data_train = _data_train.copy()
        data_eval = _data_eval.copy()
        features_train = self.extract_features(data_train)
        features_eval = self.extract_features(data_eval)
        data_train, features_train = self.align_date(data_train, features_train)
        data_eval, features_eval = self.align_date(data_eval, features_eval)

        scaler = StandardScaler()
        features_train = pd.DataFrame(scaler.fit_transform(features_train), index=features_train.index, columns=features_train.columns)
        features_eval = pd.DataFrame(scaler.transform(features_eval), index=features_eval.index, columns=features_eval.columns)
        return data_train, features_train, data_eval, features_eval

    @classmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        assert not data.isnull().sum().any()
        return data

    @classmethod
    def extract_features(self, data: pd.DataFrame, use_tech_indicator=True):
        features = pd.DataFrame(index=data.index)

        features["log_return"] = data["Close"].apply(np.log1p).diff()
        features["log_volume_diff"] = data["Volume"].apply(np.log1p).diff()
        features["volatility_20"] = features["log_return"].rolling(20).std()

        sma_20 = sma_indicator(data["Close"], window=20)
        features["ma_gap_20"] = (data["Close"] - sma_20) / sma_20

        features["high_change_20"] = (data["High"].rolling(20).max() - data["Close"]) / data["Close"]
        features["low_change_20"] = (data["Low"].rolling(20).min() - data["Close"]) / data["Close"]

        if use_tech_indicator:
            # Trend Indicators
            macd = MACD(data["Close"], window_slow=26, window_fast=12, window_sign=9)
            adx = ADXIndicator(data["High"], data["Low"], data["Close"], window=14)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                features["ADX"] = adx.adx()

            features["MACD Hist"] = macd.macd_diff()

            # Oscillator Indicator
            rsi = RSIIndicator(data["Close"], window=14).rsi()
            stoch_rsi = StochRSIIndicator(data["Close"])
            features["RSI"] = rsi
            # features["StochRSI"] = stoch_rsi.stochrsi_k() - stoch_rsi.stochrsi_d()

            # Volume Indicator
            features["MFI"] = money_flow_index(data["High"], data["Low"], data["Close"], data["Volume"], window=14)
            features["CMF"] = chaikin_money_flow(data["High"], data["Low"], data["Close"], data["Volume"])

            # Volatility Indicator
            bb = BollingerBands(data["Close"], window=20, window_dev=2)
            features["BB %B"] = bb.bollinger_pband()

        return features

    @classmethod
    def extract_features_v2(self, data: pd.DataFrame):
        features = pd.DataFrame(index=data.index)

        log_close = data["Close"].apply(np.log1p)
        log_volume = data["Volume"].apply(np.log1p)

        # Return
        features["return_1"] = log_close.diff()
        features["return_5"] = log_close.diff(5)
        # features["return_20"] = log_close.diff(20)
        # features["return_40"] = log_close.diff(40)
        # features["return_60"] = log_close.diff(60)

        # Volume
        features["volume_1"] = data["Volume"]
        features["volume_5"] = data["Volume"].rolling(5).mean()
        # features["volume_20"] = log_volume.rolling(20).mean()
        # features["volume_40"] = log_volume.rolling(40).mean()
        # features["volume_60"] = log_volume.rolling(60).mean()

        # Volatility
        features["volatility_5"] = features["return_1"].rolling(5).std()
        # features["volatility_20"] = features["return_1"].rolling(20).std()
        # features["volatility_40"] = features["return_1"].rolling(40).std()
        # features["volatility_60"] = features["return_1"].rolling(60).std()

        # Ratio of Moving Average
        features["ma_gap_5"] = (data["Close"] - (data["Close"].rolling(5).mean())) / data["Close"].rolling(5).mean()
        # features["ma_gap_20"] = log_close - (log_close.rolling(20).mean())
        # features["ma_gap_40"] = log_close - (log_close.rolling(40).mean())
        # features["ma_gap_60"] = log_close - (log_close.rolling(60).mean())
        return features.astype(np.float16)

    @classmethod
    def align_date(self, data: pd.DataFrame, features: pd.DataFrame):
        features = features.dropna(axis=0)
        data = data.loc[features.index[0] :, :].dropna(axis=0)
        assert len(data) == len(features), f"data length: {len(data)}, features length: {len(features)}"
        return data, features


def main():
    import matplotlib.pyplot as plt

    df = pd.read_csv("./data/3600/ethusd/2021-01-01.csv", parse_dates=[0]).set_index("Date") / 10

    df = Preprocessor.clean_data(df)
    features = Preprocessor.extract_features(df)
    df, features = Preprocessor.align_date(df, features)

    features.hist(figsize=(12, 10), bins=30)
    plt.show()


if __name__ == "__main__":
    main()
