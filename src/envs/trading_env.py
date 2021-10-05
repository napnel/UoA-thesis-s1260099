import numpy as np
import pandas as pd
from gym import spaces
from enum import Enum
from typing import Any, Optional, Dict, Callable
from src.envs.base_env import BaseTradingEnv, Actions
from src.envs.reward_func import profit_per_tick_reward


class DescTradingEnv(BaseTradingEnv):
    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int = 5,
        fee: float = 0.001,
        reward_func: Callable = profit_per_tick_reward,
        actions: Enum = Actions,
    ):
        super(DescTradingEnv, self).__init__(df, features, window_size, fee, reward_func)
        self.actions = actions
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.observation_size), dtype=np.float32)

    def reset(self):
        return super(DescTradingEnv, self).reset()

    def step(self, action):
        return super(DescTradingEnv, self).step(action)


class ContTradingEnv(BaseTradingEnv):
    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int = 20,
        fee: float = 0.001,
        reward_func: Callable = profit_per_tick_reward,
    ):
        super(ContTradingEnv, self).__init__(df, features, window_size, fee, reward_func)
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.observation_size), dtype=np.float32)

    def reset(self):
        # To Do: どれくらいassetsを使っているか比で表現する
        super(ContTradingEnv, self).reset()
        self.features_obs = self.features.iloc[: self.window_size, :].values
        self.account_obs = np.tile(A=[self.position.pnl_pct], reps=(self.window_size, 1))
        self.observation = np.concatenate([self.features_obs, self.account_obs], axis=0)
        return self.observation

    def step(self, action):
        self.action = action

        # Trade Start
        if self.next_done:
            self.done = True
            self.position.close()

        # actionの値から現在のequityの内どのくらいのsizeを使うか決める。
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

    @property
    def next_observation(self):
        next_features = self.features[self.current_step - self.window_size : self.current_step].values
        observation = np.concatenate(
            [self.observation[1:], np.array([next_features, self.position.pnl_pct, self.wallet.equity_pct])],
            axis=1,
        )
        return observation
