import pandas as pd
import requests
from datetime import datetime, timedelta
import os


def fetch_chart(timeframe="minute", aggregate=1, limit=30, fsym="BTC", tsym="JPY", toTs=-1) -> pd.DataFrame:
    """
    timeframe（時間軸）:「minute（1分足）」「hour(1時間足)」「day(日足)」
    limit: 取得件数(The number of data points to return)
    fsym: 通貨名(The cryptocurrency symbol of interest)
    tsym: 通貨名(The currency symbol to convert into)
    """
    url = f"https://min-api.cryptocompare.com/data/v2/histo{timeframe}"

    params = {
        "fsym": fsym,
        "tsym": tsym,
        "limit": limit,
        "aggregate": aggregate,
        "toTs": toTs,
    }

    response = requests.get(url, params, timeout=10)
    response = response.json()

    time, open, high, low, close, volume = [], [], [], [], [], []

    for data in response["Data"]["Data"]:
        time.append(datetime.fromtimestamp(data["time"]))
        open.append(data["open"])
        high.append(data["high"])
        low.append(data["low"])
        close.append(data["close"])
        volume.append(data["volumeto"] - data["volumefrom"])

    chart = pd.DataFrame({"Date": time, "Open": open, "High": high, "Low": low, "Close": close, "Volume": volume})
    return chart.set_index("Date")


def save_data(num=5, aggregate=1, fsym="BTC", tsym="USD"):
    base_dir = f"./data/{fsym}{tsym}/{aggregate}/"
    os.makedirs(base_dir, exist_ok=True)
    dt = datetime.now()
    base_date = datetime(dt.year, dt.month, dt.day)
    for i in range(num):
        date = base_date - timedelta(days=i)
        limit = (60 * 24) / aggregate
        chart = fetch_chart("minute", limit=limit, toTs=int(datetime.timestamp(date)), fsym=fsym, tsym=tsym, aggregate=aggregate)[:-1]
        chart.to_csv(f"{base_dir}test_data_{date.strftime('%Y-%m-%d')}.csv", mode="x")


def convert_timeframe(chart: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    chart: ローソク足
    rule: 期間(M, D, H, T)のいずれかを指定 ex) rule = "5T"は５分を意味する
    短い期間から長い期間に変換したい時のみ使用可能
    """
    new_chart = pd.DataFrame()
    new_chart["Open"] = chart["Open"].resample(rule=rule).first()
    new_chart["High"] = chart["High"].resample(rule=rule).max()
    new_chart["Low"] = chart["Low"].resample(rule=rule).min()
    new_chart["Close"] = chart["Close"].resample(rule=rule).last()
    return new_chart


if __name__ == "__main__":
    save_data(7, aggregate=15, fsym="ETH", tsym="USD")
