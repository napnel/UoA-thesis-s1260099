import numpy as np
from envs.trading_env import TradingEnv
from backtesting import Strategy, Backtest

from stable_baselines3 import A2C

from base import FTXAPI
from envs import TradingEnv

np.set_printoptions(suppress=True)  # 指数表記禁止


class DRLStrategy(Strategy):
    def init(self):
        self.model = A2C.load("a2c")
        self.lookback_window = 50
        self.state_size = 7
        self.env = TradingEnv(lookback_window=self.lookback_window, df=self.data.df, assets=self._broker._cash)
        self.env.reset()

    def next(self):
        print(self._broker._cash, self.env.broker.assets)
        self.step = len(self.data.df) - 1
        self.env.current_step = self.step  # BacktestのステップとEnvironmentのステップを同期させる
        
        self.state, _, done, _ = self.env.step(0)
        if self.step <= self.lookback_window:
            return

        action, _ = self.model.predict(self.state)
        if action == 0:
            pass

        elif action == 1:
            self.buy()

        elif action == 2:
            self.sell()


def main():
    ftx = FTXAPI()
    df = ftx.fetch_candle("ETH-PERP", interval=15 * 60, limit=3 * 672)
    bt = Backtest(df, DRLStrategy, cash=10000, commission=0.0, trade_on_close=True, exclusive_orders=True)
    stats = bt.run()
    bt.plot(plot_volume=False)
    print(stats)
    # print(stats["_trades"])


if __name__ == "__main__":
    main()
