from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
from ta.trend import MACD, ema_indicator
from ta.momentum import rsi

from src.utils.data_loader import DataLoader


class MACDStrategy(Strategy):
    def init(self):
        def macd_indicator(close, window_slow=26, window_fast=12, window_sign: int = 9):
            macd = MACD(close, window_slow, window_fast, window_sign)
            return macd.macd(), macd.macd_signal()

        self.macd, self.macd_signal = self.I(macd_indicator, self.data.Close.s, overlay=False, plot=True)

    def next(self):
        if crossover(self.macd, self.macd_signal):
            self.position.close()
            self.buy(size=1)

        elif crossover(self.macd_signal, self.macd):
            self.position.close()
            self.sell(size=1)


class DEMAStrategy(Strategy):
    def init(self):
        self.ema_short = self.I(ema_indicator, self.data.Close.s, window=20, overlay=True, plot=True)
        self.ema_long = self.I(ema_indicator, self.data.Close.s, window=50, overlay=True, plot=True)

    def next(self):
        if crossover(self.ema_short, self.ema_long):
            self.position.close()
            self.buy(size=1)

        elif crossover(self.ema_long, self.ema_short):
            self.position.close()
            self.sell(size=1)


def main():
    df = DataLoader.fetch_data("^N225", interval="1d", start="2018-08-31", end="2021-08-31")
    macd_bt = Backtest(df, MACDStrategy, cash=100000)
    macd_bt.run()
    macd_bt.plot()

    dema_bt = Backtest(df, DEMAStrategy, cash=100000)
    dema_bt.run()
    dema_bt.plot()


if __name__ == "__main__":
    main()
