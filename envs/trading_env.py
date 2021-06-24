import numpy as np
import pandas as pd
from functools import partial

import gym
from gym import spaces

from base import Broker


class TradingEnv(gym.Env):
    def __init__(self, lookback_window: int, df: pd.DataFrame, assets: float, fee: float = 0.0):
        self._df = df.copy()
        self._broker = partial(Broker, assets=assets, fee=fee)
        self.lookback_window = lookback_window
        self.action_space = spaces.Discrete(3)
        self.state_size = 7  # OHLCV, assets, position profit or lostt
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.lookback_window * self.state_size,))

        self.initial_assets = assets
        self.current_step: int = 0

    def reset(self):
        self.current_step = self.lookback_window
        self.begin_assets = self.initial_assets
        self.broker: Broker = self._broker(data=self._df)

        self.market_state = self.broker.get_candles(0, self.lookback_window)
        self.account_state = np.tile(self.broker.account_state, (self.lookback_window, 1))

        self.state = np.concatenate((self.market_state, self.account_state), axis=1)
        return self.state.reshape(self.lookback_window * self.state_size)

    def step(self, action):
        """
        action: 0 -> Hold, 1 -> Buy, 2 -> Sell
        """
        self.broker.current_step = self.current_step
        self.action = action
        self.current_price = self.broker.current_price

        if self.action == 0:
            pass

        elif self.action == 1:
            if self.broker.position.is_long:
                self.broker.position.close()
            else:
                size = self.broker.assets / self.current_price
                self.broker.buy(size=size, limit_price=self.current_price)

        elif self.action == 2:
            if self.broker.position.is_short:
                self.broker.position.close()
            else:
                size = self.broker.assets / self.current_price
                self.broker.sell(size=size, limit_price=self.current_price)

        self.done = False
        if self.current_step + 1 == len(self._df):
            self.broker.position.close()
            self.done = True

        self.state[:-1, :] = self.state[1:, :]
        self.state[-1, :] = np.concatenate((self.broker.latest_candle, self.broker.account_state))

        # self.market_state = self.broker.get_candles(self.current_step, self.lookback_window + self.current_step)
        # self.account_state = self.broker.account_state

        # print(self.market_state.shape, self.account_state.shape)
        # self.state = np.concatenate((self.market_state, self.account_state), axis=0).reshape(
        #     self.lookback_window * self.state_size,
        # )

        current_assets = self.broker.assets + self.broker.position.profit_or_loss
        self.reward = current_assets - self.begin_assets

        self.current_step += 1
        return self.state.reshape(self.lookback_window * self.state_size), self.reward, self.done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print(f"Assets: {self.broker.assets}")
        print(f"Action: {self.action}")
        print(f"Reward: {self.reward}, Done: {self.done}")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
