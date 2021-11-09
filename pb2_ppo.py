import os
import glob
import random
import argparse
import pandas as pd
from datetime import datetime

import ray
from ray.tune import run, sample_from
from ray.tune import CLIReporter
from ray.tune.schedulers.pb2 import PB2
from src.envs.trading_env import DescTradingEnv
from src.utils import DataLoader, Preprocessor, backtest
from src.utils.misc import get_agent_class


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="^N225")
    parser.add_argument("--max", type=int, default=30000)
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--t_ready", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
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

    data_train, features_train, data_eval, features_eval = Preprocessor.time_train_test_split(
        data,
        features,
        train_start="2012-01-01",
        train_end="2018-01-01",
        eval_start="2018-01-01",
        eval_end="2021-01-01",
        scaling=True,
    )
    user_config = {
        "env": "DescTradingEnv",
        "env_config": {
            "data": data_train,
            "features": features_train,
        },
        "model": {
            "fcnet_hiddens": [256, 64],
            # "free_log_std": True,
        },
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_num_episodes": 10,
        "evaluation_config": {
            "env_config": {
                "data": data_eval,
                "features": features_eval,
            },
            "explore": True,
        },
        "num_workers": 2,
        "framework": "torch",
        "log_level": "WARN",
        "timesteps_per_iteration": 5000,
        "num_gpus": 0,
        "seed": 3407,
        "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
        "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
    }
    ray.shutdown()
    ray.init(log_to_driver=False, num_cpus=8, num_gpus=0)
    agent_class, config = get_agent_class(args.algo)
    config.update(user_config)

    pb2 = PB2(
        time_attr=args.criteria,
        metric="evaluation/episode_reward_mean",
        mode="max",
        perturbation_interval=args.t_ready,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the hyperparam search space
        hyperparam_bounds={
            "lambda": [0.9, 1.0],
            "lr": [1e-3, 1e-5],
        },
    )

    timelog = str(datetime.date(datetime.now())) + "_" + datetime.time(datetime.now()).strftime("%H-%M")

    args.dir = "{}_{}_{}_Size{}_{}_{}".format(args.algo, args.filename, args.method, str(args.num_samples), args.env_name, args.criteria)
    reporter = CLIReporter(
        {
            "episode_reward_mean": "episode_reward",
            "evaluation/episode_reward_mean": "eval/episode_reward",
            "timesteps_total": "steps",
            "episodes_total": "episodes",
        },
        max_report_frequency=10,
    )
    analysis = run(
        agent_class,
        name="{}_{}_{}_seed{}_{}".format(timelog, args.method, args.env_name, str(args.seed), args.filename),
        scheduler=pb2,
        num_samples=4,
        progress_reporter=reporter,
        stop={args.criteria: args.max},
        config=config,
        local_dir="./experiments",
        verbose=1,
    )

    all_dfs = analysis.trial_dataframes
    names = list(all_dfs.keys())

    results = pd.DataFrame()
    for i in range(args.num_samples):
        df = all_dfs[names[i]]
        df = df[["timesteps_total", "episodes_total", "episode_reward_mean"]]
        df["Agent"] = i
        results = pd.concat([results, df]).reset_index(drop=True)
        print(results)

    if args.save_csv:
        if not (os.path.exists("data/" + args.dir)):
            os.makedirs("data/" + args.dir)

        results.to_csv("data/{}/seed{}.csv".format(args.dir, str(args.seed)))

    ray.shutdown()
