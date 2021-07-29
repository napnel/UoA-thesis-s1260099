import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ta.trend import EMAIndicator, ema_indicator
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler


def normalizing(df):
    normalized_df = df.copy()
    column_names = normalized_df.columns.tolist()
    for column in column_names:
        # Logging and Differencing
        test = np.log(normalized_df[column]) - np.log(normalized_df[column].shift(1))
        if test[1:].isnull().any():
            normalized_df[column] = normalized_df[column] - normalized_df[column].shift(1)
        else:
            normalized_df[column] = np.log(normalized_df[column]) - np.log(normalized_df[column].shift(1))
        # Min Max Scaler implemented
        min_value = normalized_df[column].min()
        max_value = normalized_df[column].max()
        normalized_df[column] = (normalized_df[column] - min_value) / (max_value - min_value)

    return normalized_df


def add_indicators(df) -> pd.DataFrame:
    indicators = pd.DataFrame()
    ema_20 = EMAIndicator(df["Close"], window=20)
    ema_50 = EMAIndicator(df["Close"], window=50)
    ema_100 = EMAIndicator(df["Close"], window=100)
    ema_200 = EMAIndicator(df["Close"], window=200)
    indicators["ema_20"] = ema_20.ema_indicator()
    indicators["ema_50"] = ema_50.ema_indicator()
    indicators["ema_100"] = ema_100.ema_indicator()
    indicators["ema_200"] = ema_200.ema_indicator()
    return indicators


def preprocessing(candles: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    price = candles["Close"]
    volume = candles["Volume"]
    ema20 = ema_indicator(price, 20)
    ema50 = ema_indicator(price, 50)
    ema200 = ema_indicator(price, 200)

    df["diff_price_pct"] = price.pct_change(1)
    df["diff_price20_pct"] = price.pct_change(20)
    df["diff_price50_pct"] = price.pct_change(50)
    df["diff_price200_pct"] = price.pct_change(200)
    df["volatility20"] = np.log(price).diff().rolling(20).std()
    df["volatility50"] = np.log(price).diff().rolling(50).std()
    df["volatility200"] = np.log(price).diff().rolling(200).std()
    df["diff_ema20_pct"] = (price - ema20) / ema20
    df["diff_ema50_pct"] = (price - ema50) / ema50
    df["diff_ema200_pct"] = (price - ema200) / ema200
    # df["diff_volume_pct"] = volume.pct_change(1)
    df["diff_log_volume"] = np.log(volume.shift(1)) - np.log(volume)
    df = df.fillna(0)
    return df


def main():
    df = pd.read_csv("./data/3600/ethusd/2021-01-01.csv", parse_dates=[0]).set_index("Date")
    print(df.head(3))

    tmp = preprocessing(df)
    print(tmp.head(50))
    tmp.hist()
    # sns.histplot(tmp)
    # tmp.hist()
    plt.show()


if __name__ == "__main__":
    main()
