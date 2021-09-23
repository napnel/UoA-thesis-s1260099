import warnings
from typing import Optional

import pandas as pd
from pandas.core.window.rolling import Window

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from backtesting import Strategy, Backtest


from ta.trend import sma_indicator, MACD
from ta.momentum import rsi
from ta.volatility import BollingerBands

RandomAgent = None


def macd_indicator(close: pd.Series, window_slow=26, window_fast=12, window_sign: int = 9):
    macd = MACD(close, window_slow, window_fast, window_sign)
    return macd.macd(), macd.macd_signal()


def bollinger_bands_indicator(close: pd.Series, window=20, window_dev=2):
    bb = BollingerBands(close, window=window, window_dev=window_dev)
    return bb.bollinger_hband(), bb.bollinger_lband()


class DRLStrategy(Strategy):
    env = None
    agent = None

    def init(self):
        self.observation = self.env.reset()
        self.done = False
        self.sma = self.I(sma_indicator, self.data.Close.s, 20, overlay=True, plot=True)
        self.macd = self.I(macd_indicator, self.data.Close.s, overlay=False, plot=True)
        self.rsi = self.I(rsi, self.data.Close.s, overlay=False, plot=True)
        self.bb = self.I(bollinger_bands_indicator, self.data.Close.s, overlay=True, plot=True)

    def next(self):
        if self.data.index[-1] != self.env.current_time or self.done:
            pass

        else:
            assert self.data.Close[-1] == self.env.closing_price, self.debug()
            assert self._broker._cash == self.env.wallet.assets, self.debug()
            assert self.equity == self.env.wallet.equity, self.debug()
            # action, _ = self.agent.predict(self.env.observation, deterministic=True)
            if self.agent:
                action = self.agent.compute_single_action(self.env.observation, explore=False)
            else:
                action = self.env.action_space.sample()
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

    def debug(self):
        print("===" * 10, "DEBUG", "===" * 10)
        print("Env Step: ", self.env.current_step)
        print("Env Price: ", self.env.closing_price, "| Backtest Price: ", self.data.Close[-1])
        print("Env Equity: ", self.env.wallet.equity, "| Backtest Equity: ", self.equity)
        print("Env Assets: ", self.env.wallet.assets, "| Backtest Assets: ", self._broker._cash)
        print("Env Position: ", self.env.position, "| Backtest Position: ", self.position)
        print("Backtest Trades: ", self.trades)
        print("===" * 10, "=====", "===" * 10)
        return "See Debug Above"


def backtest(data: pd.DataFrame, env, agent=RandomAgent, plot=False, plot_filename=None) -> pd.DataFrame:
    bt = Backtest(
        data,
        DRLStrategy,
        cash=env.wallet.initial_assets,
        commission=env.fee,
        trade_on_close=True,
    )
    stats = bt.run(env=env, agent=agent)
    if plot:
        bt.plot(filename=plot_filename)
    return stats
