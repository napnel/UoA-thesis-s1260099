import warnings
from typing import Optional

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from backtesting import Strategy, Backtest
from stable_baselines3.common.base_class import BaseAlgorithm

from .utils import get_action_prob


class DRLStrategy(Strategy):
    model: Optional[BaseAlgorithm] = None
    env = None

    def init(self):
        self.observation = self.env.reset()
        self.step = 0
        self.max_step = len(self.data.df) - 2
        self.done = False
        self.history_action_prob = np.zeros(self.max_step)

    def next(self):
        if self.step < self.env.current_step or self.done:
            pass

        else:
            assert self.data.Close[-1] == self.env.closing_price, self.debug()
            assert self._broker._cash == self.env.wallet.assets, self.debug()
            assert self.equity == self.env.wallet.equity, self.debug()
            action, _ = self.model.predict(self.env.observation, deterministic=True)
            # do Trade
            if self.env.next_done:
                self.position.close()

            elif action == self.env.actions.Buy.value and not self.position.is_long:
                if self.position.is_short:
                    self.position.close()
                else:
                    self.buy(size=1)

            elif action == self.env.actions.Sell.value and not self.position.is_short:
                if self.position.is_long:
                    self.position.close()
                else:
                    self.sell(size=1)

            self.observation, _, self.done, _ = self.env.step(action)

        self.step = len(self.data.df)

    def debug(self):
        print("===" * 10, "DEBUG", "===" * 10)
        print("Env Step: ", self.env.current_step, "Backtest Step: ", self.step)
        print("Env Price: ", self.env.closing_price, "Backtest Price: ", self.data.Close[-1])
        print("Env Equity: ", self.env.wallet.equity, "Backtest Equity: ", self.equity)
        print("Env Assets: ", self.env.wallet.assets, "Backtest Assets: ", self._broker._cash)
        print("Env Position: ", self.env.position, "Backtest Position: ", self.position)
        print("Backtest Trades: ", self.trades)
        print("===" * 10, "=====", "===" * 10)
        return "See Debug Above"


def backtest(model: BaseAlgorithm, env, plot=False, plot_filename=None) -> pd.DataFrame:
    bt = Backtest(
        env.df,
        DRLStrategy,
        cash=env.wallet.initial_assets,
        commission=env.fee,
    )
    stats = bt.run(model=model, env=env)
    if plot:
        bt.plot(filename=plot_filename)
    return stats
