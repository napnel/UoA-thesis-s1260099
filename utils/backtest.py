import warnings
import pandas as pd
from typing import Optional

from bokeh.util.warnings import BokehDeprecationWarning

warnings.simplefilter("ignore", BokehDeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
from backtesting import Strategy, Backtest
from stable_baselines3.common.base_class import BaseAlgorithm


class DRLStrategy(Strategy):
    model: Optional[BaseAlgorithm] = None
    env = None

    def init(self):
        self.state = self.env.reset()
        self.max_step = len(self.data.df) - 1

    def next(self):
        self.step = len(self.data.df) - 1
        self.env.current_step = self.step  # BacktestのステップとEnvironmentのステップを同期させる

        if self.step < self.env.window_size:
            return

        assert self._broker._cash == self.env.wallet.assets, f"Step:{self.step}/{self.max_step}: {self._broker._cash} != {self.env.wallet.assets}"
        assert self.equity == self.env.wallet.equity, f"Step{self.step}/{self.max_step}: {self.equity} != {self.env.wallet.equity}"

        if self.step == self.max_step:
            self.env.position.close()
            return

        action, _ = self.model.predict(self.env.next_observation, deterministic=True)

        if action == self.env.actions.Buy.value and not self.position.is_long:
            if self.position.is_short:
                self.position.close()
            else:
                self.buy()
            self.env.buy()

        elif action == self.env.actions.Sell.value and not self.position.is_short:
            if self.position.is_long:
                self.position.close()
            else:
                self.sell()
            self.env.sell()


def backtest(model: BaseAlgorithm, env, plot=False, plot_filename=None) -> pd.DataFrame:
    bt = Backtest(env._df, DRLStrategy, cash=env.wallet.initial_assets, commission=env.fee, trade_on_close=True, exclusive_orders=False)
    stats = bt.run(model=model, env=env)
    if plot:
        bt.plot(filename=plot_filename)
    return stats
