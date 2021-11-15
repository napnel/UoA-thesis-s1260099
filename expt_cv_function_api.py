import os
import pathlib
import glob
import random
import json
import argparse
from numpy import isin
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch

import ray
from ray import tune
from ray.tune import Callback
from ray.tune import ProgressReporter
from ray.tune import CLIReporter
from ray.tune.logger import UnifiedLogger
from ray.tune.schedulers.pb2 import PB2

from src.envs import BaseTradingEnv
from src.utils import DataLoader, Preprocessor, backtest
from src.utils.misc import get_agent_class
from ray.rllib.agents import dqn, a3c, ppo, sac, ddpg
from src.trainable.cross_validation import ExperimentCV


def experiment_cv(config, checkpoint_dir=None):
    ticker = config.pop("_ticker")
    n_splits = config.pop("_n_splits")
    algo = config.pop("_algo")
    max_timesteps = config.pop("_max_timesteps")

    if len(glob.glob(f"./data/{ticker}/*.csv")) != 0:
        data = DataLoader.load_data(f"./data/{ticker}/ohlcv.csv")
        features = DataLoader.load_data(f"./data/{ticker}/features.csv")
    else:
        os.makedirs(f"./data/{ticker}", exist_ok=True)
        data = DataLoader.fetch_data(f"{ticker}", interval="1d")
        data, features = Preprocessor.extract_features(data)
        data.to_csv(f"./data/{ticker}/ohlcv.csv")
        features.to_csv(f"./data/{ticker}/features.csv")

    agent_class, algo_config = get_agent_class(algo)
    algo_config.update(config)
    agent = agent_class(config=algo_config)
    # agent.reset()

    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            state = json.loads(f.read())
            accuracy = state["acc"]
            start = state["step"]

    if checkpoint_dir:
        print("Loading from checkpoint.")
        agent.restore(checkpoint_path)
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint["step"]

    agents = [None] * n_splits
    summary_results = []

    for i, (data_train, features_train, data_eval, features_eval) in enumerate(
        Preprocessor.blocked_cross_validation(data, features, n_splits=n_splits, train_years=1)
    ):
        algo_config["env_config"]["data"] = data_train
        algo_config["env_config"]["features"] = features_train
        algo_config["evaluation_config"]["env_config"]["data"] = data_eval
        algo_config["evaluation_config"]["env_config"]["features"] = features_eval

        if checkpoint_dir:
            pass

        agents[i] = agent_class(
            config=algo_config, logger_creator=lambda config: UnifiedLogger(config, os.path.join(tune.get_trial_dir(), f"splits_{i}"))
        )
        # agents[i] = agent_class(config=algo_config)

        history_results = []

        while True:
            results = agents[i].train()
            if results["timesteps_total"] > max_timesteps:
                break

            history_results.append(results)
            # tune.report(**results)
            # if step % 3 == 0:
            #     with tune.checkpoint_dir(step=step) as checkpoint_dir:
            #         path = os.path.join(checkpoint_dir, "checkpoint")
            #         with open(path, "w") as f:
            #             f.write(json.dumps({"acc": accuracy, "step": start}))

            # if step % 5 == 0:
            #     # Every 5 steps, checkpoint our current state.
            #     # First get the checkpoint directory from tune.
            #     with tune.checkpoint_dir(step=step) as checkpoint_dir:
            #         # Then create a checkpoint file in this directory.
            #         path = os.path.join(checkpoint_dir, "checkpoint")
            #         # Save state to checkpoint file.
            #         # No need to save optimizer for SGD.
            #         torch.save({"step": step, "model_state_dict": model.state_dict(), "mean_accuracy": acc}, path)
            #         agent.save()
            # agents[i].save()

        summary_results.append(history_results)
        agents[i].stop()

    for column in range(len(summary_results[0])):
        for row in range(len(summary_results)):
            results = summary_results[row][column]

    print("Trial Dir: ", tune.get_trial_dir())
    print("Trial Id: ", tune.get_trial_id())
    print("Trial Name: ", tune.get_trial_name())
    print("Resorces: ", tune.get_trial_resources())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Basic Settings
    parser.add_argument("--ticker", type=str, default="^N225")
    parser.add_argument("--algo", type=str, default="A2C")
    # parser.add_argument("--algo", type=str, nargs="*", default="DQN")
    parser.add_argument("--max_timesteps", type=int, default=5000)
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
        "num_workers": 4,
        "framework": "torch",
        "log_level": "WARN",
        "timesteps_per_iteration": 2500,
        "num_gpus": 0,
        "seed": args.seed,
        "_algo": args.algo,
        "_ticker": args.ticker,
        "_n_splits": 5,
        "_max_timesteps": args.max_timesteps,
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
        experiment_cv,
        # ExperimentCV,
        name=f"{args.algo}_{timelog}",
        num_samples=args.num_samples,
        metric=args.metric,
        mode=args.mode,
        # stop={"timesteps_total": args.max_timesteps},
        config=config,
        progress_reporter=reporter,
        # checkpoint_freq=1,
        local_dir=args.logdir,
        resources_per_trial=tune.PlacementGroupFactory([{"CPU": 4}, {"CPU": 4}]),
        # resources_per_trial=dqn.DQNTrainer.default_resource_request()
        # scheduler=pb2,
        # reuse_actors=True,
        # verbose=1,
    )

    print(analysis.best_config)
    print(analysis.best_checkpoint)
    # trainer = ExperimentCV(config=analysis.best_config)
    # trainer.restore(analysis.best_checkpoint)
    # log_dir = pathlib.Path(analysis.best_checkpoint).parent
    # for i, agent in enumerate(trainer.agents):
    #     # env_train = agent.workers.local_worker().env
    #     env_eval = agent.evaluation_workers.local_worker().env
    #     # backtest(env_train, agent, save_dir=os.path.join(log_dir, f"splits_{i}", "last-stats-train"), plot=False)
    #     backtest(env_eval, agent, save_dir=os.path.join(log_dir, f"splits_{i}", "last-stats-eval"), plot=True)

    all_dfs = analysis.trial_dataframes
    print(all_dfs)

    # analysis.get_best_checkpoint()
    ray.shutdown()
