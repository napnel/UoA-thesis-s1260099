import os
import glob
from datetime import datetime
from threading import local
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
from ray.tune.logger import UnifiedLogger, CSVLogger, JsonLogger, TBXLogger

from src.envs.trading_env import DescTradingEnv
from src.envs.reward_func import equity_log_return_reward, initial_equity_return_reward
from src.utils import DataLoader, Preprocessor, backtest
from src.utils.logger import custom_log_creator, create_log_filename
from src.utils.misc import clean_result, clean_stats, get_agent_class, send_line_notification


def experiment(config, checkpoint_dir=None):
    stop_timesteps_total = config.pop("_timesteps_total")
    n_splits = config.pop("_n_splits")

    agent_class: Trainer = config.pop("_agent_class")
    data: pd.DataFrame = config.pop("_data")
    features: pd.DataFrame = config.pop("_features")
    window_size: int = config.pop("_window_size")
    reward_func = config.pop("_reward_func")

    if args.algo == "DQN":
        config["hiddens"] = [64, 16]

    config["env_config"] = {"reward_func": reward_func, "window_size": window_size}
    config["evaluation_config"]["env_config"] = {"reward_func": reward_func, "window_size": window_size}

    results_list = []
    for i, (data_train, features_train, data_eval, features_eval) in enumerate(
        Preprocessor.blocked_cross_validation(data, features, n_splits=n_splits)
    ):
        config["env_config"]["data"] = data_train
        config["env_config"]["features"] = features_train
        config["evaluation_config"]["env_config"]["data"] = data_eval
        config["evaluation_config"]["env_config"]["features"] = features_eval
        agent = agent_class(config=config, logger_creator=lambda config: UnifiedLogger(config, f"{i}"))

        history_results = []

        while True:
            train_results = agent.train()
            if train_results["timesteps_total"] > stop_timesteps_total:
                break

            history_results.append(train_results)
            tune.report(**train_results)

        # for results in history_results:
        #     tune.report(**results)

        agent.save()
        # checkpoint = agent.save(f"{tune.get_trial_dir()}_{i}")
        del agent
        # agent.stop()

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

    if len(glob.glob(f"./data/{args.ticker}/*.csv")) != 0:
        data = DataLoader.load_data(f"./data/{args.ticker}_ohlcv.csv")
        features = DataLoader.load_data(f"./data/{args.ticker}_features.csv")
    else:
        os.makedirs(f"./data/{args.ticker}", exist_ok=True)
        data = DataLoader.fetch_data(args.ticker, interval="1d")
        data, features = Preprocessor.extract_features(data)
        data.to_csv(f"./data/{args.ticker}/ohlcv.csv")
        features.to_csv(f"./data/{args.ticker}/features.csv")

    ray.shutdown()
    ray.init(log_to_driver=False, num_cpus=4, num_gpus=0)
    agent_class, config = get_agent_class(args.algo)

    experiment_name = f"{args.algo}_{datetime.date(datetime.now())}_{datetime.time(datetime.now())}".split(".")[0].replace(":", "-")
    print(experiment_name)

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
            "fcnet_hiddens": [64, 64],
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
        "_timesteps_total": 10000,
        "_agent_class": agent_class,
        "_data": data,
        "_features": features,
        "_window_size": 5,
        "_reward_func": equity_log_return_reward,
        "_n_splits": 2,
    }
    config.update(user_config)

    analysis = tune.run(
        experiment,
        config=config,
        resources_per_trial=agent_class.default_resource_request(config),
        progress_reporter=reporter,
        local_dir="./experiments",
        name=experiment_name,
        verbose=1,
        reuse_actors=True,
    )

    all_dfs = analysis.trial_dataframes
    print(all_dfs)

    send_line_notification("Lab | Training Finished")
    ray.shutdown()
