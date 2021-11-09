import os
import glob
import random
import argparse
from numpy import isin
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.logger import UnifiedLogger
from ray.tune.schedulers.pb2 import PB2
from src.envs.trading_env import DescTradingEnv
from src.utils import DataLoader, Preprocessor, backtest
from src.utils.misc import get_agent_class


class Trainer(tune.Trainable):
    def setup(self, config):
        n_splits = config.pop("_n_splits")
        agent_class = config.pop("_agent_class")
        data = config.pop("_data")
        features = config.pop("_features")
        self.agents = []
        for i, (data_train, features_train, data_eval, features_eval) in enumerate(
            Preprocessor.blocked_cross_validation(data, features, n_splits=n_splits)
        ):
            config["env_config"]["data"] = data_train
            config["env_config"]["features"] = features_train
            config["evaluation_config"]["env_config"]["data"] = data_eval
            config["evaluation_config"]["env_config"]["features"] = features_eval
            self.agents.append(agent_class(config=config, logger_creator=lambda config: UnifiedLogger(config, f"{i}")))

    def step(self):
        averaged_results = {}
        for i, agent in enumerate(self.agents):
            results = agent.train()
            averaged_results = self.average_results(i, results, averaged_results)

        return averaged_results

    def save_checkpoint(self, tmp_checkpoint_dir):
        for agent in self.agents:
            agent.save()
        return tmp_checkpoint_dir

    def average_results(self, n_total: int, results: Dict[str, Any], avg_res: Optional[dict]):
        partial_avg_res = avg_res if avg_res else {}

        for key, value in results.items():
            if isinstance(value, (int, float)):
                partial_avg_res[key] = (n_total * partial_avg_res[key] + value) / (n_total + 1) if partial_avg_res.get(key) else value
            elif isinstance(value, dict):
                partial_avg_res[key] = self.average_results(n_total, value, partial_avg_res.get(key))

        return partial_avg_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="^N225")
    parser.add_argument("--max", type=int, default=50000)
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--t_ready", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturb", type=float, default=0.5)  # if using PBT
    parser.add_argument("--env_name", type=str, default="DescTradingEnv")
    parser.add_argument("--criteria", type=str, default="timesteps_total")  # "training_iteration", "time_total_s"
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--save_csv", type=bool, default=False)
    parser.add_argument("--method", type=str, default="pb2")
    args = parser.parse_args()

    if len(glob.glob("./data/*.csv")) != 0:
        data = DataLoader.load_data(f"./data/{args.ticker}_ohlcv.csv")
        features = DataLoader.load_data(f"./data/{args.ticker}_features.csv")
    else:
        os.makedirs("./data", exist_ok=True)
        data = DataLoader.fetch_data(args.ticker, interval="1d")
        data, features = Preprocessor.extract_features(data)
        data.to_csv(f"./data/{args.ticker}_ohlcv.csv")
        features.to_csv(f"./data/{args.ticker}_features.csv")

    ray.shutdown()
    ray.init(log_to_driver=False, num_gpus=0)
    agent_class, config = get_agent_class(args.algo)

    user_config = {
        "env": "DescTradingEnv",
        "env_config": {},
        "model": {
            "fcnet_hiddens": [64, 64],
        },
        # "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_num_episodes": 10,
        "evaluation_config": {
            "env_config": {},
            "explore": True,
        },
        "num_workers": 1,
        "framework": "torch",
        "log_level": "INFO",
        "timesteps_per_iteration": 5000,
        "num_gpus": 0,
        "seed": 3407,
        "_agent_class": agent_class,
        "_data": data,
        "_features": features,
        "_n_splits": 5,
    }
    config.update(user_config)

    timelog = str(datetime.date(datetime.now())) + "_" + datetime.time(datetime.now()).strftime("%H-%M")

    reporter = CLIReporter(
        {
            "episode_reward_mean": "episode_reward",
            "evaluation/episode_reward_mean": "eval/episode_reward",
            "timesteps_total": "steps",
            "episodes_total": "episodes",
        },
        max_report_frequency=10,
    )
    analysis = tune.run(
        Trainer,
        name=f"{args.algo}_{timelog}_{args.env_name}",
        num_samples=args.num_samples,
        progress_reporter=reporter,
        stop={args.criteria: args.max},
        config=config,
        local_dir="./experiments",
        checkpoint_at_end=True,
        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 4}, {"CPU": 8}]),
        verbose=1,
    )

    all_dfs = analysis.trial_dataframes
    print(all_dfs)
    names = list(all_dfs.keys())

    # results = pd.DataFrame()
    # for i in range(args.num_samples):
    #     df = all_dfs[names[i]]
    #     df = df[["timesteps_total", "episodes_total", "episode_reward_mean"]]
    #     df["Agent"] = i
    #     results = pd.concat([results, df]).reset_index(drop=True)
    #     print(results)

    # if args.save_csv:
    #     if not (os.path.exists("data/" + args.dir)):
    #         os.makedirs("data/" + args.dir)

    #     results.to_csv("data/{}/seed{}.csv".format(args.dir, str(args.seed)))

    ray.shutdown()
