import warnings
import numpy as np
import pandas as pd
from ta.trend import sma_indicator, MACD, ADXIndicator, STCIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import money_flow_index, chaikin_money_flow


class Preprocessor:
    @classmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        assert not data.isnull().sum().any()
        return data

    @classmethod
    def extract_features(self, data: pd.DataFrame):
        features = pd.DataFrame(index=data.index)

        # Trend Indicators
        macd = MACD(data["Close"], window_slow=26, window_fast=12, window_sign=9)
        adx = ADXIndicator(data["High"], data["Low"], data["Close"], window=14)
        stc = STCIndicator(data["Close"])
        features["DMI_Diff"] = adx.adx_pos() - adx.adx_neg()
        features["MACD"] = macd.macd_diff()
        features["STC"] = stc.stc()

        # Oscillator Indicator
        rsi = RSIIndicator(data["Close"], window=14).rsi()
        stoch_rsi = StochRSIIndicator(data["Close"])
        features["RSI"] = rsi
        features["StochRSI"] = stoch_rsi.stochrsi_k() - stoch_rsi.stochrsi_d()

        # Volume Indicator
        features["MFI"] = money_flow_index(data["High"], data["Low"], data["Close"], data["Volume"], window=14)
        features["CMF"] = chaikin_money_flow(data["High"], data["Low"], data["Close"], data["Volume"])
        return features

    @classmethod
    def extract_features_v2(self, data: pd.DataFrame):
        features = pd.DataFrame(index=data.index)

        log_close = data["Close"].apply(np.log1p)
        log_volume = data["Volume"].apply(np.log1p)

        # Return
        features["return_1"] = log_close.diff()
        features["return_5"] = log_close.diff(5)
        features["return_20"] = log_close.diff(20)
        features["return_40"] = log_close.diff(40)
        features["return_60"] = log_close.diff(60)

        # Volume
        features["volume_1"] = log_volume
        features["volume_5"] = log_volume.rolling(5).mean()
        features["volume_20"] = log_volume.rolling(20).mean()
        features["volume_40"] = log_volume.rolling(40).mean()
        features["volume_60"] = log_volume.rolling(60).mean()

        # Volatility
        features["volatility_5"] = features["return_1"].rolling(5).std()
        features["volatility_20"] = features["return_1"].rolling(20).std()
        features["volatility_40"] = features["return_1"].rolling(40).std()
        features["volatility_60"] = features["return_1"].rolling(60).std()

        # Ratio of Moving Average
        features["ma_gap_5"] = log_close - (log_close.rolling(5).mean())
        features["ma_gap_20"] = log_close - (log_close.rolling(20).mean())
        features["ma_gap_40"] = log_close - (log_close.rolling(40).mean())
        features["ma_gap_60"] = log_close - (log_close.rolling(60).mean())
        return features.astype(np.float16)

    @classmethod
    def align_date(self, data: pd.DataFrame, features: pd.DataFrame):
        features = features.dropna(axis=0)
        data = data.loc[features.index[0] :, :].dropna(axis=0)
        assert len(data) == len(features), f"data length: {len(data)}, features length: {len(features)}"
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
