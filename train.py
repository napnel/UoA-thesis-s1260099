import numpy as np
import pandas as pd
import warnings
import os

from envs import TradingEnv
from utils import load_data, preprocessing


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C, PPO


def evaluate(model: A2C, env, render=True):
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
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    # ftx = FTXAPI()
    # df = ftx.fetch_candle("ETH-PERP", interval=15 * 60, limit=3 * 672)
    df = load_data("./data/ETHUSD/15", limit_days=5)
    preprocessed_df = preprocessing(df)
    lookback_window = 50
    env = TradingEnv(lookback_window=lookback_window, df=df, preprocessed_df=preprocessed_df, assets=10000, fee=0.0001)
    check_env(env, warn=True)
    env = make_vec_env(lambda: env, n_envs=1)
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir).learn(total_timesteps=30000)
    model.save("a2c")
    # model = PPO("MlpPolicy", env, verbose=1).learn(total_timesteps=30000)
    # model.save("ppo")
    # mean_reward, _ = evaluate_policy(model, env, render=False)
    mean_reward, std_reward = evaluate(model, env, render=True)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")


if __name__ == "__main__":
    main()
