import os
import pathlib
import glob
import random
import argparse
from numpy import isin
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

import ray
from ray import tune
from ray.tune import Callback
from ray.tune import ProgressReporter
from ray.tune import CLIReporter
from ray.tune.logger import UnifiedLogger
from ray.tune.schedulers.pb2 import PB2
from src.envs import BaseTradingEnv
from src.utils import DataLoader, Preprocessor, backtest
from src.trainable.cross_validation import ExperimentCV


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Basic Settings
    parser.add_argument("--ticker", type=str, default="^N225")
    parser.add_argument("--algo", type=str, default="DQN")
    # parser.add_argument("--algo", type=str, nargs="*", default="DQN")
    parser.add_argument("--max_timesteps", type=int, default=30000)
    parser.add_argument("--metric", type=str, default="evaluation/episode_reward_mean")
    parser.add_argument("--mode", type=str, default="max")
    # Hyperparameter Tuning Settings
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--criteria", type=str, default="timesteps_total")
    parser.add_argument("--perturb", type=float, default=0.25)
    # Other Settings
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--logdir", type=str, default="./experiments")
    args = parser.parse_args()
    print(args)

    ray.shutdown()
    ray.init(log_to_driver=False, num_gpus=0, local_mode=False)

    config = {
        "env": "BaseTradingEnv",
        "env_config": {},
        # "model": {
        #     "fcnet_hiddens": [64, 64],
        # },
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "env_config": {},
            "explore": False,
        },
        "num_workers": 1,
        "framework": "torch",
        "log_level": "WARN",
        "timesteps_per_iteration": 5000,
        "num_gpus": 0,
        "seed": args.seed,
        "_algo": args.algo,
        "_ticker": args.ticker,
        "_n_splits": 4,
        # "lambda": tune.sample_from(lambda spec: random.uniform(0.9, 1.0)),
        # "lr": tune.sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
    }

    pb2 = PB2(
        time_attr=args.criteria,
        perturbation_interval=2500,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the hyperparam search space
        hyperparam_bounds={
            "lr": [1e-3, 1e-5],
        },
    )

    timelog = str(datetime.date(datetime.now())) + "_" + datetime.time(datetime.now()).strftime("%H-%M")

    reporter = CLIReporter(
        {
            "episode_reward_mean": "episode_reward",
            "evaluation/episode_reward_mean": "eval/episode_reward",
            "timesteps_total": "steps",
            "episodes_total": "episodes",
        },
        max_report_frequency=20,
    )
    analysis = tune.run(
        ExperimentCV,
        name=f"{args.algo}_{timelog}",
        num_samples=args.num_samples,
        metric=args.metric,
        mode=args.mode,
        stop={"timesteps_total": args.max_timesteps},
        config=config,
        progress_reporter=reporter,
        checkpoint_freq=1,
        local_dir=args.logdir,
        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 4}, {"CPU": 4}]),
        # scheduler=pb2,
        # reuse_actors=True,
        verbose=1,
    )

    print(analysis.best_config)
    trainer = ExperimentCV(config=analysis.best_config)
    trainer.restore(analysis.best_checkpoint)
    log_dir = pathlib.Path(analysis.best_checkpoint).parent
    for i, agent in enumerate(trainer.agents):
        # env_train = agent.workers.local_worker().env
        env_eval = agent.evaluation_workers.local_worker().env
        # backtest(env_train, agent, save_dir=os.path.join(log_dir, f"splits_{i}", "last-stats-train"), plot=False)
        backtest(env_eval, agent, save_dir=os.path.join(log_dir, f"splits_{i}", "best-stats-eval"), plot=True)
        backtest(env_eval, agent="Buy&Hold", save_dir=os.path.join(log_dir, f"splits_{i}", "buy-and-hold"))

    all_dfs = analysis.trial_dataframes
    print(all_dfs)
    # analysis.get_best_checkpoint()
    ray.shutdown()
