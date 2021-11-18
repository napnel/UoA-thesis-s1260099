import os
import pathlib
import glob
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional

import ray
from ray import tune
from ray.tune.suggest.repeater import Repeater
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune import CLIReporter
from src.envs import BaseTradingEnv
from src.utils import backtest
from src.utils.misc import prepare_config_for_agent

from src.trainable.cross_validation_repeater import ExperimentCV


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
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--criteria", type=str, default="timesteps_total")
    parser.add_argument("--perturb", type=float, default=0.25)
    # Other Settings
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--local_dir", type=str, default="./experiments")
    args = parser.parse_args()
    print(args)

    ray.shutdown()
    ray.init(log_to_driver=False, num_gpus=0, local_mode=False)

    config = {
        "env": "BaseTradingEnv",
        "env_config": {},
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "env_config": {},
            "explore": False,
        },
        "num_workers": 4,
        "framework": "torch",
        "log_level": "WARN",
        "timesteps_per_iteration": 5000,
        "num_gpus": 0,
        "seed": args.seed,
        "_algo": args.algo,
        "_ticker": args.ticker,
        "_train_start": "2010-01-01",
        "_train_years": 5,
        "_eval_years": 1
        # "lambda": tune.sample_from(lambda spec: random.uniform(0.9, 1.0)),
        # "lr": tune.sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
    }
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

    re_searcher = Repeater(BayesOptSearch(), repeat=args.repeat)

    analysis = tune.run(
        ExperimentCV,
        name=f"{args.algo}_{timelog}",
        num_samples=args.repeat * args.num_samples,
        metric=args.metric,
        mode=args.mode,
        stop={"timesteps_total": args.max_timesteps},
        config=config,
        progress_reporter=reporter,
        checkpoint_freq=1,
        local_dir=args.local_dir,
        trial_dirname_creator=lambda trial: str(trial).split("__")[0],
        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 4}, {"CPU": 4}]),
        search_alg=re_searcher,
        verbose=1,
    )

    # Backtest
    all_config = analysis.get_all_configs()
    agent = None
    for trial in analysis.trials:
        config = all_config[trial.logdir].copy()
        agent_class, algo_config = prepare_config_for_agent(config, pathlib.Path(trial.logdir).parent)

        algo_config["num_workers"] = 1
        algo_config["logger_config"] = {"type": ray.tune.logger.NoopLogger}
        # index = config.pop("__trial_index__")

        if agent is None:
            agent = agent_class(config=algo_config)
        else:
            agent.reset(algo_config)

        checkpoint = analysis.get_best_checkpoint(trial)
        agent.restore(checkpoint)

        # env_train = agent.workers.local_worker().env
        env_train = BaseTradingEnv(**algo_config["env_config"])
        env_eval = BaseTradingEnv(**algo_config["evaluation_config"]["env_config"])
        print(backtest(env_train, agent, save_dir=os.path.join(trial.logdir, "last-stats-train"), plot=False))
        print(backtest(env_eval, agent, save_dir=os.path.join(trial.logdir, "best-stats-eval"), plot=True))
        backtest(env_eval, agent="Buy&Hold", save_dir=os.path.join(trial.logdir, "buy-and-hold"), plot=False)

    ray.shutdown()
