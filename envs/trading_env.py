from math import copysign
import sys
import numpy as np
import pandas as pd
from functools import partial

import gym
from gym import spaces

from .broker import Broker


class TradingEnv(gym.Env):
    def __init__(self, lookback_window: int, df: pd.DataFrame, preprocessed_df: pd.DataFrame, assets: float, fee: float = 0.0):
        self._df = df.copy()
        self._preprocessed_df = preprocessed_df
        self._broker = partial(Broker, assets=assets, fee=fee)
        self.lookback_window = lookback_window
        self.action_space = spaces.Discrete(3)
        self.state_size = len(self._preprocessed_df.columns) + 2  # OHLCV, assets, position profit or loss
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.lookback_window * self.state_size,))

        self.initial_assets = assets
        self.current_step: int = 0

    def reset(self):
        self.current_step = self.lookback_window
        self.prev_assets = self.initial_assets
        self.broker: Broker = self._broker(data=self._df)

        self.market_state = self._preprocessed_df.iloc[: self.lookback_window, :]
        self.account_state = np.tile(self.broker.account_state, (self.lookback_window, 1))

        self._state = np.concatenate((self.market_state, self.account_state), axis=1)
        return self.state

    def step(self, action):
        """
        action: 0 -> Hold, 1 -> Buy, 2 -> Sell
        """
        self.broker.current_step = self.current_step
        self.action = action

        if self.action == 0:
            pass

        elif self.action == 1 and not self.broker.position.is_long:
            self.buy()

        elif self.action == 2 and not self.broker.position.is_short:
            self.sell()

        self.done = self.is_terminal
        if self.done:
            self.broker.position.close()

        self.update_state()

        current_assets = self.broker.assets + self.broker.position.profit_or_loss
        self.reward = (current_assets - self.prev_assets) / self.prev_assets

        self.prev_assets = current_assets
        self.current_step += 1
        return self.state, self.reward, self.done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print(f"Assets: {self.broker.assets}")
        print(f"Action: {self.action}")
        print(f"Reward: {self.reward}, Done: {self.done}")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def buy(self):
        if self.broker.position.size == 0:
            size = self.broker.free_assets // (self.broker.current_price * (1 + self.broker.fee))
            self.broker.new_order(size, self.broker.current_price)
        elif self.broker.position.size < 0:  # If you have position, then positon close
            self.broker.new_order(-self.broker.position.size, self.broker.current_price)

    def sell(self):
        if self.broker.position.size == 0:
            size = self.broker.free_assets // (self.broker.current_price * (1 + self.broker.fee))
            self.broker.new_order(-size, self.broker.current_price)
        elif self.broker.position.size > 0:  # If you have position, then positon close
            self.broker.new_order(-self.broker.position.size, self.broker.current_price)

    def update_state(self):
        self._state[:-1, :] = self._state[1:, :]
        self._state[-1, :] = np.concatenate((self._preprocessed_df.iloc[self.current_step, :], self.broker.account_state))

    @property
    def is_terminal(self):
        return True if self.current_step + 1 == len(self._df) or self.broker.assets < self.broker.current_price else False

    @property
    def state(self) -> np.ndarray:
        return self._state.reshape(self.lookback_window * self.state_size)
