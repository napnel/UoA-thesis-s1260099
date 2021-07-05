import numpy as np
from yaml import load
from envs.trading_env import TradingEnv
from backtesting import Strategy, Backtest
from ta.trend import sma_indicator
from stable_baselines3 import A2C

from broker import FTXAPI
from envs import TradingEnv
from utils import load_data

np.set_printoptions(suppress=True)  # 指数表記禁止


class DRLStrategy(Strategy):
    def init(self):
        self.model = A2C.load("a2c")
        self.lookback_window = 200
        self.state_size = 7
        self.env = TradingEnv(lookback_window=self.lookback_window, df=self.data.df, assets=self._broker._cash)
        self.state = self.env.reset()
        self.action = 0
        self.done = False
        self.sma = self.I(sma_indicator, self.data.Close.s, 14)
        self.max_step = len(self.data.df)

    def next(self):
        self.step = len(self.data.df) - 1
        self.env.current_step = self.step  # BacktestのステップとEnvironmentのステップを同期させる
        self.env.broker.current_step = self.step
        self.env.update_state()

        if self.step < self.lookback_window:
            return

        if self.step == self.max_step:
            self.env.broker.position.close()
            self.position.close()
            return

        print("=" * 50, self.step, "=" * 50)
        assert self._broker._cash, self.env.broker.assets
        assert self.equity == self.env.broker.equity
        print(self._broker.position, self.env.broker.position)
        print(self._broker.trades)
        print("Assets:", self._broker._cash, self.env.broker.assets)
        print("Equity:", self.equity, self.env.broker.equity)
        print("Close Price:", self._broker.last_price, self.env.broker.current_price)
        print("State: ", self.env.state[-self.state_size :])
        print("Done", self.env.is_terminal)

        self.action, _ = self.model.predict(self.env.state)
        if self.action == 0:
            pass

        elif self.action == 1 and not self.position.is_long:
            print("BUY")
            if self.position.is_short:
                self.position.close()
            else:
                self.buy()
            self.env.buy()

        elif self.action == 2 and not self.position.is_short:
            print("SELL")
            if self.position.is_long:
                self.position.close()
            else:
                self.sell()
            self.env.sell()


def main():
    ftx = FTXAPI()
    df = load_data("./data/ETHUSD/15", 5)
    # df = ftx.fetch_candle("ETH-PERP", interval=15 * 60, limit=800)
    bt = Backtest(df, DRLStrategy, cash=10000, commission=0.0, trade_on_close=True, exclusive_orders=False)
    stats = bt.run()
    # bt.plot(plot_volume=False)
    print(stats)
    print(stats["_trades"])
    print(stats["_strategy"].env.broker.closed_trades)
    print(df.tail(3))


if __name__ == "__main__":
    main()
