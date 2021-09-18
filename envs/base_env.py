import gym
import numpy as np
import pandas as pd
from gym import spaces
from math import copysign
from enum import Enum
from typing import Any, Optional, Dict, Callable

from .reward_func import profit_per_tick_reward


class Actions(Enum):
    Sell = 0
    Buy = 1
    # Neutral = 2


class Position:
    def __init__(self, env: "BaseTradingEnv"):
        self.__env = env
        self.size: int = 0
        self.entry_price: float = 0.0
        self.entry_time: Optional[pd.Timestamp] = None

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
        return self.size * (self.__env.opening_price - self.entry_price)

    @property
    def profit_or_loss_pct(self) -> float:
        if self.size == 0:
            return 0
        return copysign(1, self.size) * (self.__env.opening_price - self.entry_price) / self.entry_price

    def close(self):
        if self.size == 0:
            return
        self.__env.wallet.assets += self.profit_or_loss
        trade = {
            "Steps": self.__env.current_step,
            "Size": self.size,
            "EntryPrice": self.entry_price,
            "ExitPrice": self.__env.opening_price,
            "ReturnPct": self.profit_or_loss_pct,
            "EntryTime": self.entry_time,
            "ExitTime": self.__env.current_datetime,
        }
        self.__env.closed_trades = self.__env.closed_trades.append(trade, ignore_index=True)
        self.size = 0
        self.entry_price = 0.0
        self.entry_time = None


class Wallet:
    def __init__(self, env: "BaseTradingEnv", assets: Optional[float] = None):
        self.__env = env
        self.initial_assets = 10 ** len(str(self.__env.df["High"].max()).split(".")[0]) if assets is None else assets
        self.assets = self.initial_assets

    @property
    def equity(self) -> float:
        return self.assets + self.__env.position.size * (self.__env.closing_price - self.__env.position.entry_price)

    @property
    def equity_pct(self) -> float:
        return (self.equity - self.initial_assets) / (self.initial_assets)

    @property
    def free_assets(self) -> float:
        used_assets = abs(self.__env.position.size) * self.__env.closing_price
        return max(0, self.equity - used_assets)


class BaseTradingEnv(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int = 20,
        fee: float = 0.001,
        reward_func: Callable = profit_per_tick_reward,
    ):
        """ """
        self.df = df.copy()
        self.features = features.copy()
        self.fee = fee
        self.window_size = window_size
        self.reward_func = reward_func
        self.current_step = 0
        self.position: Optional[Position] = Position(self)
        self.wallet: Optional[Wallet] = Wallet(self)
        self.closed_trades: Optional[pd.DataFrame] = None

        self.action_space = None
        self.observation_space = None

    def reset(self):
        self.done = False
        self.next_done = False
        self.current_step = self.window_size
        self.position = Position(self)
        self.wallet = Wallet(self)
        self.observation = self.features.iloc[: self.window_size, :].values
        self.closed_trades = pd.DataFrame(columns=["Steps", "Size", "EntryPrice", "ExitPrice", "ReturnPct", "EntryTime", "ExitTime"])
        return self.observation

    def step(self, action):
        self.action = action

        # Trade Start
        if self.next_done:
            self.done = True
            self.position.close()

        elif self.action == Actions.Buy.value and not self.position.is_long:
            self.buy(size=1)

        elif self.action == Actions.Sell.value and not self.position.is_short:
            self.sell(size=1)

        # Trade End

        self.current_step += 1
        self.next_done = True if self.current_step >= len(self.df) - 3 else False

        self.observation = self.next_observation
        self.reward = self.reward_func(self)
        self.info = {}

        return self.observation, self.reward, self.done, self.info

    def render(self, mode="human"):
        print("===" * 10, f"Step: {self.current_step}", "===" * 10)
        print(f"Equity: {self.wallet.equity}")
        print(f"Position: {self.position.size}, {self.position.profit_or_loss}")
        print(f"Action: {self.action}, Reward: {self.reward}, Done: {self.done}")

    def buy(self, size: Optional[float] = None):
        if self.position.size == 0:
            adjusted_price = self.opening_price * (1 + self.fee)
            self.position.size = int(self.wallet.free_assets // adjusted_price) if size is None else size
            self.position.entry_price = adjusted_price
            self.position.entry_time = self.current_datetime

        elif self.position.is_short:
            self.position.close()

    def sell(self, size: Optional[float] = None):
        if self.position.size == 0:
            adjusted_price = self.opening_price * (1 - self.fee)
            self.position.size = -int(self.wallet.free_assets // adjusted_price) if size is None else -size
            self.position.entry_price = adjusted_price
            self.position.entry_time = self.current_datetime

        elif self.position.is_long:
            self.position.close()

    @property
    def next_observation(self):
        return self.features[self.current_step - self.window_size : self.current_step].values

    @property
    def closing_price(self):
        return self.df["Close"][self.current_step]

    @property
    def opening_price(self):
        return self.df["Open"][self.current_step + 1]

    @property
    def current_datetime(self):
        return self.df.index[self.current_step + 1]

    @property
    def show_tech_indicators(self):
        pass
