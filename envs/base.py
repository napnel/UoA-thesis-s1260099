from argparse import Action
import gym
import numpy as np
import pandas as pd
from gym import spaces
from math import copysign
from enum import Enum
from typing import Optional


class Actions(Enum):
    Sell = 0
    Buy = 1


class Position:
    def __init__(self, env: gym.Env):
        self.__env = env
        self.size: int = 0
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[float] = None

    def __repr__(self) -> str:
        return f"Position(size: {self.size}, entry_price: {self.entry_price}, pl: {self.profit_or_loss:.0f})"

    @property
    def is_long(self) -> bool:
        return True if self.size > 0 else False

    @property
    def is_short(self) -> bool:
        return True if self.size < 0 else False

    @property
    def profit_or_loss(self) -> float:
        if self.size == 0:
            return 0
        return self.size * (self.__env.current_price - self.entry_price)

    @property
    def profit_or_loss_pct(self) -> float:
        if self.size == 0:
            return 0
        return copysign(1, self.size) * (self.__env.current_price - self.entry_price) / self.entry_price

    def close(self):
        if self.size == 0:
            return
        self.__env.wallet.assets += self.profit_or_loss
        trade = {
            "Size": self.size,
            "EntryPrice": self.entry_price,
            "ExitPrice": self.__env.current_price,
            "PnL": self.profit_or_loss,
            "ReturnPct": self.profit_or_loss_pct,
            "EntryTime": self.entry_time,
            "ExitTime": self.__env.current_datetime,
        }
        self.__env.closed_trades = self.__env.closed_trades.append(trade, ignore_index=True)
        self.size = 0
        self.entry_price = None
        self.entry_time = None


class Wallet:
    def __init__(self, env: gym.Env, assets: float = 1000000):
        self.__env = env
        self.initial_assets = assets
        self.assets = assets

    @property
    def equity(self) -> float:
        return self.assets + self.__env.position.profit_or_loss

    @property
    def free_assets(self) -> float:
        used_assets = abs(self.__env.position.size) * self.__env.current_price
        return max(0, self.equity - used_assets)


class BaseTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, preprocessed_df: pd.DataFrame, window_size: int, fee: float, actions: Enum = Actions):
        self._df = df.copy()
        self._preprocessed_df = preprocessed_df
        self.fee = fee
        self.window_size = window_size
        self.actions = actions
        self.action_space = spaces.Discrete(len(actions))
        self.observation_size = len(self._preprocessed_df.columns)  # positionの情報を持つか検討する
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.observation_size))
        self.current_step = 0
        self.position: Optional[Position] = None
        self.wallet: Optional[Wallet] = None
        self.closed_trades: Optional[pd.DataFrame] = None

    def reset(self):
        self.current_step = self.window_size
        self.position = Position(self)
        self.wallet = Wallet(self)
        self.observation = self._preprocessed_df.iloc[: self.window_size, :].values
        self.prev_profit_or_loss_pct = 0
        self.prev_equity = self.wallet.equity
        self.closed_trades = pd.DataFrame(columns=["Size", "EntryPrice", "ExitPrice", "PnL", "ReturnPct", "EntryTime", "ExitTime"])
        return self.observation

    def step(self, action):
        self.action = action

        if self.action == Actions.Buy.value and not self.position.is_long:
            self.buy()

        elif self.action == Actions.Sell.value and not self.position.is_short:
            self.sell()

        self.done = True if self.current_step + 1 == len(self._df) else False
        if self.done:
            self.position.close()

        self.observation = self.next_observation
        self.reward = self._calculate_reward()
        self.info = {}

        self.prev_profit_or_loss_pct = self.position.profit_or_loss_pct
        self.prev_equity = self.wallet.equity
        self.current_step += 1
        return self.observation, self.reward, self.done, self.info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print(f"Assets: {self.wallet.assets}, Equity: {self.wallet.equity}")
        print(f"Position: {self.position.size}, {self.position.profit_or_loss}")
        print(f"Action: {self.action}, Reward: {self.reward}, Done: {self.done}")
        print(self.closed_trades.tail(1))

    def buy(self):
        if self.position.size == 0:
            adjusted_price = self.current_price * (1 + self.fee)
            self.position.size = int(self.wallet.free_assets // adjusted_price)
            self.position.entry_price = adjusted_price
            self.position.entry_time = self.current_datetime

        elif self.position.is_short:
            self.position.close()

    def sell(self):
        if self.position.size == 0:
            adjusted_price = self.current_price * (1 - self.fee)
            self.position.size = -int(self.wallet.free_assets // adjusted_price)
            self.position.entry_price = adjusted_price
            self.position.entry_time = self.current_datetime

        elif self.position.is_long:
            self.position.close()

    def _calculate_reward(self):
        raise NotImplementedError()

    @property
    def next_observation(self):
        return self._preprocessed_df[self.current_step - self.window_size + 1 : self.current_step + 1].values

    @property
    def current_price(self):
        return self._df["Close"][self.current_step]

    @property
    def current_datetime(self):
        return self._df.iloc[self.current_step, :].name


class SimpleTradingEnv(BaseTradingEnv):
    def __init__(self, df: pd.DataFrame, preprocessed_df: pd.DataFrame, window_size: int, fee: float):
        super().__init__(df, preprocessed_df, window_size, fee)

    def _calculate_reward(self):
        reward = self.position.profit_or_loss_pct - self.prev_profit_or_loss_pct * 100
        self.prev_profit_or_loss_pct = self.position.profit_or_loss_pct
        return reward
