# Backtestingのモジュールを使う必要ないかも

import numpy as np
import pandas as pd
from functools import partial

import gym
from gym import spaces

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from ftx_api import FTXAPI
from broker import Broker


class TradingEnv(gym.Env):
    def __init__(self, lookback_widow_size: int, df: pd.DataFrame, assets: float, fee: float = 0.0):
        self._df = df.copy()
        self._broker = partial(Broker, assets=assets, fee=fee)
        self.lookback_widow_size = lookback_widow_size
        self.action_space = spaces.Discrete(3)
        self.state_size = 5  # Open, High, Low, Close, Volume
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.lookback_widow_size * self.state_size,))

        self.initial_assets = assets
        self.current_step: int = 0

    def reset(self):
        self.current_step = 0
        self.begin_assets = self.initial_assets
        self.broker: Broker = self._broker(data=self._df)

        self.state = self.broker.get_candles(0, self.lookback_widow_size).reshape(
            self.lookback_widow_size * self.state_size,
        )
        return self.state

    def step(self, action):
        """
        action: 0 -> Hold, 1 -> Buy, 2 -> Sell
        """
        self.current_step += 1
        self.broker.current_index = self.lookback_widow_size + self.current_step
        self.action = action
        self.current_price = self.broker.current_price

        if self.action == 0:
            pass

        elif self.action == 1:
            self.broker.position.close()
            self.broker.buy(size=1, limit_price=self.current_price)

        elif self.action == 2:
            self.broker.position.close()
            self.broker.sell(size=1, limit_price=self.current_price)

        self.state = self.broker.get_candles(self.current_step, self.lookback_widow_size + self.current_step).reshape(
            self.lookback_widow_size * self.state_size,
        )

        current_assets = self.broker.assets + self.broker.position.profit_or_loss
        self.reward = current_assets - self.begin_assets

        self.done = False
        if self.lookback_widow_size + self.current_step + 1 == len(self._df):
            self.done = True

        return self.state, self.reward, self.done, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}")
        print(f"Assets: {self.broker.assets}")
        print(f"Action: {self.action}")
        print(f"Reward: {self.reward}, Done: {self.done}")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


def evaluate(model: A2C, env: VecEnv, render=True):
    state = env.reset()
    done = False
    episode_rewards = []
    episode_reward = 0.0

    while not done:
        action, state = model.predict(state)
        state, reward, done, info = env.step(action)

        episode_reward += reward

        if render:
            env.render()

        episode_rewards.append(episode_reward)

        if done:
            break

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def main():
    ftx = FTXAPI()
    df = ftx.fetch_candle("ETH-PERP", interval=15 * 60, limit=3 * 672)
    windows_size = 50
    env = TradingEnv(lookback_widow_size=windows_size, df=df, assets=10000)
    check_env(env, warn=True)
    env = make_vec_env(lambda: env, n_envs=1)
    model = A2C("MlpPolicy", env, verbose=1).learn(total_timesteps=10000)
    # mean_reward, _ = evaluate_policy(model, env, render=False)
    mean_reward, std_reward = evaluate(model, env, render=True)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")


if __name__ == "__main__":
    main()
