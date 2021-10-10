import gym
import numpy as np
import pandas as pd
from math import copysign
from enum import Enum
from typing import Any, Optional, Dict, Callable

from src.envs.reward_func import equity_log_return_reward


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
        self.stop_loss: Optional[float] = -1

    def __repr__(self) -> str:
        return f"Position(size: {self.size}, entry_price: {self.entry_price}, PnL: {self.pnl:.0f})"

    @property
    def is_long(self) -> bool:
        return True if self.size > 0 else False

    @property
    def is_short(self) -> bool:
        return True if self.size < 0 else False

    @property
    def pnl(self) -> float:
        if self.size == 0:
            return 0
        return self.size * (self.__env.closing_price - self.entry_price)

    @property
    def pnl_pct(self) -> float:
        if self.size == 0:
            return 0
        return copysign(1, self.size) * (self.__env.closing_price - self.entry_price) / self.entry_price

    def close(self, stop=False):
        if self.size == 0:
            return

        pnl = self.pnl if not stop else self.size * (self.stop_loss - self.entry_price)
        self.__env.wallet.assets += pnl
        trade = {
            "Steps": self.__env.current_step,
            "Size": self.size,
            "EntryPrice": self.entry_price,
            "ExitPrice": self.__env.closing_price if not stop else self.stop_loss,
            "ReturnPct": self.pnl_pct if not stop else copysign(1, self.size) * (self.stop_loss - self.entry_price) / self.entry_price,
            "EntryTime": self.entry_time,
            "ExitTime": self.__env.current_time,
        }
        self.__env.closed_trades = self.__env.closed_trades.append(trade, ignore_index=True)
        self.size = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = None


class Wallet:
    def __init__(self, env: "BaseTradingEnv", assets: Optional[float] = None):
        self.__env = env
        self.initial_assets = 100000
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
        window_size: int = 5,
        fee: float = 0.001,
        reward_func: Callable = equity_log_return_reward,
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
        self.observation_size = len(self.features.columns) + 1

        self.action_space = None
        self.observation_space = None

    def reset(self):
        self.done = False
        self.next_done = False
        self.reward = 0
        self.current_step = self.window_size
        self.position = Position(self)
        self.wallet = Wallet(self)
        self.equity_curve = [self.wallet.equity]
        self.closed_trades = pd.DataFrame(columns=["Steps", "Size", "EntryPrice", "ExitPrice", "ReturnPct", "EntryTime", "ExitTime"])
        features_obs = self.features.iloc[: self.window_size, :].values
        account_obs = np.tile([self.position.pnl_pct], (self.window_size, 1))
        self.observation = np.hstack((features_obs, account_obs))
        return self.observation

    def step(self, action):
        self.action = action

        # Trade Start
        if self.next_done:
            self.done = True
            self.position.close()

        elif self.action == Actions.Buy.value and not self.position.is_long:
            self.buy()
            # self.buy(size=1)
            # self.buy(stop_loss=self.df["Close"][self.current_step - self.window_size : self.current_step].min() * (1 - 0.005))

        elif self.action == Actions.Sell.value and not self.position.is_short:
            self.sell()
            # self.sell(size=1)
            # self.sell(stop_loss=self.df["Close"][self.current_step - self.window_size : self.current_step].max() * (1 + 0.005))

        # Trade End

        self.current_step += 1

        self.judge_stop_loss()

        self.equity_curve.append(self.wallet.equity)
        self.next_done = True if self.current_step >= len(self.df) - 3 else False

        self.observation = self.next_observation
        self.reward = self.reward_func(self)
        self.info = {}

        return self.observation, self.reward, self.done, self.info

    def render(self, mode="human"):
        print("===" * 10, f"Step: {self.current_step}", "===" * 10)
        print(f"Equity: {self.wallet.equity}")
        print(f"Position: {self.position.size}, {self.position.pnl}")
        print(f"Action: {self.action}, Reward: {self.reward}, Done: {self.done}")

    def buy(self, size: Optional[float] = None, stop_loss: Optional[float] = None):
        if self.position.size == 0:
            adjusted_price = self.closing_price * (1 + self.fee)
            self.position.size = int(self.wallet.free_assets // adjusted_price) if size is None else size
            self.position.entry_price = adjusted_price
            self.position.entry_time = self.current_time
            self.position.stop_loss = stop_loss

        elif self.position.is_short:
            self.position.close()

    def sell(self, size: Optional[float] = None, stop_loss: Optional[float] = None):
        if self.position.size == 0:
            adjusted_price = self.closing_price * (1 - self.fee)
            self.position.size = -int(self.wallet.free_assets // adjusted_price) if size is None else -size
            self.position.entry_price = adjusted_price
            self.position.entry_time = self.current_time
            self.position.stop_loss = stop_loss

        elif self.position.is_long:
            self.position.close()

    def judge_stop_loss(self):
        if not self.position.stop_loss:
            return

        if self.position.is_short and self.df["Open"][self.current_step] <= self.position.stop_loss <= self.df["Close"][self.current_step]:
            self.position.close(stop=True)

        elif self.position.is_long and self.df["Open"][self.current_step] >= self.position.stop_loss >= self.df["Close"][self.current_step]:
            self.position.close(stop=True)

    @property
    def next_observation(self):
        next_single_obs = np.hstack((self.features.iloc[self.current_step - 1, :], self.position.pnl_pct))
        next_observation = np.vstack((self.observation[1:], next_single_obs))
        return next_observation

    @property
    def closing_price(self):
        return self.df["Close"][self.current_step]

    @property
    def current_time(self):
        return self.df.index[self.current_step]

    @property
    def tech_indicators(self):
        return self.features.columns.tolist()
