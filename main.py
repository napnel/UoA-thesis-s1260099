import argparse
import os
import pathlib
from datetime import datetime

import dill as pickle
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.stopper import MaximumIterationStopper
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.repeater import Repeater
from ray.tune.trial import Trial

from src.evaluation import backtest_expt
from src.experiments import ExperimentCV
from src.tuning_space import get_tuning_params

parser = argparse.ArgumentParser()
# Basic Settings
parser.add_argument("--local_dir", type=str, required=True)
parser.add_argument("--ticker", type=str, default="^N225")
parser.add_argument("--algo", type=str, default="DQN")
parser.add_argument("--max_iter", type=int, default=10)

# Environment Settings (Optional)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--fee", type=float, default=None)
parser.add_argument("--reward_func", type=str, default=None)
parser.add_argument("--actions", type=str, default=None)
parser.add_argument("--stop_loss", action="store_true")

# Cross-validation Settings
parser.add_argument("--repeat", type=int, default=5)
parser.add_argument("--train_years", type=int, default=5)
parser.add_argument("--eval_years", type=int, default=2)
parser.add_argument("--num_samples", type=int, default=30)

# Other Settings
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--expt_name", type=str, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
print(args)

assert args.num_samples > 1, "Can't tune them."

METRIC = "evaluation/episode_reward_mean"
MODE = "max"


def main(config):
    parameter_columns = ["__trial_index__"]
    tuning_params = get_tuning_params(args.algo)
    for param, bounds in tuning_params.items():
        if "env_config/" in param:
            config["env_config"][param.split("/")[-1]] = bounds
        else:
            config[param] = bounds
        parameter_columns.append(param)

    searcher_alg = Repeater(
        HyperOptSearch(metric=METRIC, mode=MODE), repeat=args.repeat
    )
    stopper = MaximumIterationStopper(max_iter=args.max_iter)

    reporter = CLIReporter(
        {
            "episode_reward_mean": "episode_reward",
            "evaluation/episode_reward_mean": "eval/episode_reward",
            "timesteps_total": "steps",
            "episodes_total": "episodes",
        },
        parameter_columns=parameter_columns,
        max_progress_rows=30,
        max_report_frequency=30,
    )

    # Create experimental name
    timelog = (
        str(datetime.date(datetime.now()))
        + "_"
        + datetime.time(datetime.now()).strftime("%H-%M")
    )
    expt_name = (
        f"{args.algo}__{timelog}"
        if args.expt_name is None
        else f"{args.algo}__{args.expt_name}__{timelog}"
    )
    print(expt_name)

    def trial_dirname_creator(trial: Trial):
        return (
            f"{trial.trainable_name}-{trial.config['__trial_index__']}_{trial.trial_id}"
        )

    analysis = tune.run(
        ExperimentCV,
        name=expt_name,
        num_samples=args.repeat * args.num_samples,
        metric=METRIC,
        mode=MODE,
        stop=stopper,
        config=config,
        progress_reporter=reporter,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
        trial_dirname_creator=trial_dirname_creator,
        resources_per_trial=tune.PlacementGroupFactory([{}, {"CPU": args.num_workers}]),
        search_alg=searcher_alg,
        verbose=1,
    )

    with open(os.path.join(args.local_dir, expt_name, "analysis.pkl"), "wb") as f:
        pickle.dump(analysis, f)

    print(f"This experiment is saved at {pathlib.Path(analysis.best_logdir).parent}")

    backtest_expt(analysis)


if __name__ == "__main__":
    ray.shutdown()
    ray.init(num_gpus=0)
    config = {
        "env": "TradingEnv",
        "env_config": {
            "window_size": args.window_size,
            "fee": args.fee,
            "actions": args.actions,
            "reward_func": args.reward_func,
            "stop_loss": args.stop_loss,
        },
        "model": {
            "fcnet_hiddens": [256, 128],
            "fcnet_activation": "relu",
            "custom_model": "batch_norm_model",
        },
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "env_config": {},
            "explore": False,
        },
        "num_workers": args.num_workers,
        "framework": "torch",
        "log_level": "WARN" if not args.debug else "DEBUG",
        "timesteps_per_iteration": 5000,
        "num_gpus": 0,
        "seed": args.seed,
        "_algo": args.algo,
        "_ticker": args.ticker,
        "_cv_config": {
            "train_start": "2010-01-01",
            "train_years": args.train_years,
            "eval_years": args.eval_years,
        },
        "_env_test_config": {},
    }
    main(config)
    ray.shutdown()
