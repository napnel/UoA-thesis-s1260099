import os
import glob
from datetime import datetime
from time import time
import pandas as pd
import shutil
import random
import argparse
from pprint import pprint

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.tune import CLIReporter
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents import ppo, dqn
from ray.rllib.models import ModelCatalog

from src.envs.trading_env import DescTradingEnv
from src.envs.reward_func import equity_log_return_reward, initial_equity_return_reward
from src.utils import DataLoader, Preprocessor, backtest
from src.utils.logger import custom_log_creator, create_log_filename
from src.utils.misc import clean_result, clean_stats, get_agent_class, send_line_notification


def experiment(config, checkpoint_dir=None):
    stop_timesteps_total = config.pop("_timesteps_total")

    agent_class: Trainer = config.pop("_agent_class")
    data: pd.DataFrame = config.pop("_data")
    features: pd.DataFrame = config.pop("_features")
    window_size: int = config.pop("_window_size")
    reward_func = config.pop("_reward_func")

    data_train, features_train, data_eval, features_eval = Preprocessor.train_test_split(
        data,
        features,
        train_start="2015-01-01",
        train_end="2020-01-01",
        eval_start="2020-01-01",
        eval_end="2021-09-30",
        scaling=True,
    )

    config["env_config"] = {
        "df": data_train,
        "features": features_train,
        "reward_func": reward_func,
        "window_size": window_size,
    }
    config["evaluation_config"]["env_config"] = {
        "df": data_eval,
        "features": features_eval,
        "reward_func": reward_func,
        "window_size": window_size,
    }

    train_agent: Trainer = agent_class(config=config)
    checkpoint = None
    train_results = {}

    # print("===" * 10, "Train", "===" * 10)
    while True:
        train_results = train_agent.train()
        if train_results["timesteps_total"] > stop_timesteps_total:
            break

        tune.report(**train_results)

    checkpoint = train_agent.save(tune.get_trial_dir())
    # checkpoint = train_agent.save()
    # print(checkpoint)
    # print("===" * 10, "Train", "===" * 10)
    train_agent.stop()

    # Manual Eval
    # config["num_workers"] = 0
    # config["evaluation_num_workers"] = 0
    # agent: Trainer = agent_class(config=config)
    # print("===" * 10, "Eval", "===" * 10)
    # env_train = agent.workers.local_worker().env
    # env_eval = agent.evaluation_workers.local_worker().env
    # agent.restore(checkpoint)
    # print(agent.workers.local_worker().env)
    # print(agent.evaluation_workers.local_worker().env)
    # print(agent.logdir)
    # print("===" * 10, "Eval", "===" * 10)

    # backtest(env_train, agent, save_dir=os.path.join(agent.logdir, "last-stats-train"), plot=True)
    # backtest(env_eval, agent, save_dir=os.path.join(agent.logdir, "last-stats-eval"), plot=True)
    # obs = env.reset()
    # done = False
    # eval_results = {"eval_reward": 0, "eval_eps_length": 0}
    # while not done:
    #     action = eval_agent.compute_single_action(obs)
    #     next_obs, reward, done, info = env.step(action)
    #     eval_results["eval_reward"] += reward
    #     eval_results["eval_eps_length"] += 1
    # results = {**train_results, **eval_results}
    # tune.report(**results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN", "A2C", "PPO", "SAC"])
    parser.add_argument("--ticker", type=str, default="^N225")
    parser.add_argument("--expt-name", type=str, default=None)
    args = parser.parse_args()

    ray.shutdown()
    ray.init(log_to_driver=False, num_gpus=0)

    if len(glob.glob(f"./data/{args.ticker}/*.csv")) != 0:
        data = DataLoader.load_data(f"./data/{args.ticker}_ohlcv.csv")
        features = DataLoader.load_data(f"./data/{args.ticker}_features.csv")
    else:
        os.makedirs(f"./data/{args.ticker}", exist_ok=True)
        data = DataLoader.fetch_data(args.ticker, interval="1d")
        data, features = Preprocessor.extract_features(data)
        data.to_csv(f"./data/{args.ticker}/ohlcv.csv")
        features.to_csv(f"./data/{args.ticker}/features.csv")

    agent_class, config = get_agent_class(args.algo)

    timelog = f"{datetime.date(datetime.now())}_{datetime.time(datetime.now())}".split(".")[0].replace(":", "-")
    print(timelog)

    reporter = CLIReporter(
        {
            "episode_reward_mean": "episode_reward",
            "evaluation/episode_reward_mean": "eval/episode_reward",
            "timesteps_total": "steps",
            "episodes_total": "episodes",
        },
        max_report_frequency=10,
    )

    user_config = {
        "env": "DescTradingEnv",
        "model": {
            "fcnet_hiddens": [256, 64],
        },
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_num_episodes": 5,
        "num_workers": 2,
        "framework": "torch",
        "log_level": "WARN",
        "timesteps_per_iteration": 5000,
        "num_gpus": 0,
        "seed": 3407,
        "_timesteps_total": 50000,
        "_agent_class": agent_class,
        "_data": data,
        "_features": features,
        "_window_size": 9,
        "_reward_func": equity_log_return_reward,
    }
    config.update(user_config)

    analysis = tune.run(
        experiment,
        config=config,
        resources_per_trial=agent_class.default_resource_request(config),
        progress_reporter=reporter,
        local_dir="./experiments",
        name=timelog,
        verbose=1,
    )

    all_dfs = analysis.trial_dataframes
    print(all_dfs)

    ray.shutdown()
    send_line_notification("Lab | Training Finished")
