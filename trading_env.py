import numpy as np
import pandas as pd
from functools import partial

import gym
from gym import spaces
from backtesting.backtesting import _Broker
from backtesting._util import _Data

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from ftx_api import FTXAPI


class TradingEnv(gym.Env):
    def __init__(
        self,
        window_size: int,
        df: pd.DataFrame,
        cash: float,
        commission: float = 0.0,
        margin: float = 1.0,
        trade_on_close: bool = False,
        hedging: bool = False,
        exclusive_orders: bool = False,
    ):
        self._df = df.copy()
        self._broker = partial(
            _Broker,
            cash=cash,
            commission=commission,
            margin=margin,
            trade_on_close=trade_on_close,
            hedging=hedging,
            exclusive_orders=exclusive_orders,
            index=self._df.index,
        )
        self.window_size = window_size
        self.action_space = spaces.Discrete(3)
        self.state_size = 5
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.window_size * self.state_size,))

        self.current_step: int = 0

    def reset(self):
        data = _Data(self._df.copy(deep=False))
        self.broker: _Broker = self._broker(data=data)
        self.current_step = 0
        self.coin_bought = 0.0

        state = self.broker._data.df.iloc[: self.window_size].values
        state = state.reshape(
            self.window_size * self.state_size,
        )
        return state

    def step(self, action):
        """
        action: 0 -> Hold, 1 -> Buy, 2 -> Sell
        """
        # self.price = self.df.iloc[self.window_size + self.current_step, :].values
        # self.price = np.squeeze(self.price)

        obs = self.broker._data.df.iloc[self.current_step : self.window_size + self.current_step].values
        obs = obs.reshape(
            self.window_size * self.state_size,
        )

        if action == 0:
            pass

        elif action == "buy":
            self.broker.new_order(1)

        elif action == "sell":
            self.broker.new_order(1)

        reward = self.broker.position.pl

        done = False
        if self.window_size + self.current_step + 1 == len(self._df):
            done = True

        info = {}
        self.current_step += 1

        return obs, reward, done, info

    def render(self, mode="human"):
        print(f"Total Step: {self.current_step}")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


def main():
    ftx = FTXAPI()
    df = ftx.fetch_candle("ETH-PERP", interval=15 * 60, limit=5 * 672)
    windows_size = 50
    max_step_size = len(df) - windows_size
    env = TradingEnv(window_size=windows_size, df=df, cash=10000)
    check_env(env, warn=True)
    env = make_vec_env(lambda: env, n_envs=1)
    model = A2C("MlpPolicy", env, verbose=1).learn(max_step_size)
    mean_reward, _ = evaluate_policy(model, env)
    print(mean_reward)


if __name__ == "__main__":
    main()
