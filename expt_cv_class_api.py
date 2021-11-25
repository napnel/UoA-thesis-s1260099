import os
import pathlib
import argparse
from datetime import datetime

import ray
from ray import tune
from ray.tune.suggest.repeater import Repeater
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune import CLIReporter
from src.envs import TradingEnv
from src.envs.actions import LongNeutralShort, BuySell
from src.utils import backtest
from src.trainable.cross_validation import ExperimentCV
from src.trainable.util import prepare_config_for_agent


parser = argparse.ArgumentParser()
# Basic Settings
parser.add_argument("--ticker", type=str, default="^N225")
parser.add_argument("--algo", type=str, default="DQN")
parser.add_argument("--max_timesteps", type=int, default=30000)
parser.add_argument("--metric", type=str, default="evaluation/episode_reward_mean")
parser.add_argument("--mode", type=str, default="max")

# Environment Settings (Optional)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--fee", type=float, default=None)
parser.add_argument("--reward_func", type=str, default=None)
parser.add_argument("--actions", type=str, default=None)

# Cross-validation Settings
parser.add_argument("--repeat", type=int, default=5)
parser.add_argument("--train_years", type=int, default=5)
parser.add_argument("--eval_years", type=int, default=1)

# Hyperparameter Tuning Settings (To do)
parser.add_argument("--num_samples", type=int, default=1)

# Other Settings
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--local_dir", type=str, default="./experiments")
parser.add_argument("--expt_name", type=str, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
print(args)


if __name__ == "__main__":
    ray.shutdown()
    ray.init(log_to_driver=False, num_gpus=0, local_mode=args.debug)

    config = {
        "env": "TradingEnv",
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
        "timesteps_per_iteration": 10000,
        "num_gpus": 0,
        "seed": args.seed,
        "_algo": args.algo,
        "_ticker": args.ticker,
        # Cross-validation Settings
        "_train_start": "2010-01-01",
        "_train_years": args.train_years,
        "_eval_years": args.eval_years,
        # Environment Optional Settings
        "_window_size": args.window_size,
        "_fee": args.fee,
        "_actions": args.actions,
        "_reward_func": args.reward_func,
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
        max_report_frequency=60,
    )

    re_searcher = Repeater(BayesOptSearch(), repeat=args.repeat)

    # def expt_name_creator(args):
    args_dict = vars(args)
    expt_name = {}
    # pass

    analysis = tune.run(
        ExperimentCV,
        name=f"{args.algo}_{timelog}_{args.actions}",  # algo_EnvSetting
        num_samples=args.repeat * args.num_samples,
        metric=args.metric,
        mode=args.mode,
        stop={"timesteps_total": args.max_timesteps},
        config=config,
        progress_reporter=reporter,
        checkpoint_freq=1,
        local_dir=args.local_dir,
        trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.config['__trial_index__']}",  # trial index -> target periods
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
        env_train = TradingEnv(**algo_config["env_config"])
        env_eval = TradingEnv(**algo_config["evaluation_config"]["env_config"])
        backtest(env_train, agent, save_dir=os.path.join(trial.logdir, "last-stats-train"), plot=True, open_browser=args.debug)
        backtest(env_eval, agent, save_dir=os.path.join(trial.logdir, "best-stats-eval"), plot=True)
        backtest(env_eval, agent="Buy&Hold", save_dir=os.path.join(trial.logdir, "buy-and-hold"), plot=False)

    print(f"This experiment is saved at {pathlib.Path(analysis.best_logdir).parent}")
    ray.shutdown()
