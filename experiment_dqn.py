import os
import glob
import shutil
import argparse
from pprint import pprint

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.suggest.variant_generator import grid_search

from src.envs.trading_env import DescTradingEnv
from src.envs.reward_func import equity_log_return_reward
from src.utils import DataLoader, Preprocessor, backtest
from src.utils.logger import custom_log_creator, create_log_filename
from src.utils.misc import clean_result, clean_stats, get_agent_class, send_line_notification


def main(args):
    if len(glob.glob("./data/*.csv")) != 0:
        data = DataLoader.load_data(f"./data/{args.ticker}_ohlcv.csv")
        features = DataLoader.load_data(f"./data/{args.ticker}_features.csv")
    else:
        os.makedirs("./data", exist_ok=True)
        data = DataLoader.fetch_data(args.ticker, interval="1d")
        data, features = Preprocessor.extract_features(data)
        data.to_csv(f"./data/{args.ticker}_ohlcv.csv")
        features.to_csv(f"./data/{args.ticker}_features.csv")

    data_train, features_train, data_eval, features_eval = Preprocessor.train_test_split(
        data,
        features,
        train_start="2009-01-01",
        train_end="2019-01-01",
        eval_start="2019-01-01",
        eval_end="2021-01-01",
        scaling=True,
    )

    reward_func = equity_log_return_reward
    user_config = {
        "env": "DescTradingEnv",
        "env_config": {
            "df": data_train,
            "features": features_train,
            "reward_func": reward_func,
            "window_size": args.window_size,
        },
        "evaluation_num_workers": 1,
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "env_config": {
                "df": data_eval,
                "features": features_eval,
                "reward_func": reward_func,
                "window_size": args.window_size,
            },
            "explore": False,
        },
        "num_workers": 4,
        "framework": "torch",
        "log_level": "WARN",
        "num_gpus": 0,
        "seed": 3407,
    }

    agent_class, config = get_agent_class(args.algo)
    config.update(user_config)

    if args.algo == "DQN":
        config["prioritized_replay"] = tune.grid_search([False, True])
        config["double_q"] = tune.grid_search([False, True])
        config["dueling"] = tune.grid_search([False, True])
        config["noisy"] = tune.grid_search([False, True])
        config["n_step"] = tune.grid_search([1, 5])
        config["num_atoms"] = tune.grid_search([1, 200])

    analysis = tune.run(
        agent_class,
        config=config,
        local_dir=args.logdir,
        stop={"timesteps_total": args.stop_timesteps},
        name=args.expt_name,
        checkpoint_at_end=True,
    )

    best_config = analysis.get_best_config(metric="evaluation/episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_last_checkpoint(metric="evaluation/episode_reward_mean", mode="max")
    agent = agent_class(config=best_config)
    agent.reset_config(best_config)
    agent.restore(best_checkpoint)
    print(agent.get_policy().model)

    # env_train = DescTradingEnv(**config["env_config"])
    # backtest(env_train, agent, plot=False, plot_filename=os.path.join(agent.logdir, "backtest_train"))
    env_eval = DescTradingEnv(**config["evaluation_config"]["env_config"])
    backtest(env_eval, agent, save_dir=os.path.join(agent.logdir, "last-stats-eval"), plot=True)
    print("best checkpoint saved at", best_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="DQN", choices=["DQN", "A2C", "PPO", "SAC"])
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--ticker", type=str, default="^N225")
    parser.add_argument("--stop-timesteps", type=int, default=int(3e5))
    parser.add_argument("--logdir", type=str, default="./ray_results")
    parser.add_argument("--expt-name", type=str, default=None)

    # DQN Options
    parser.add_argument("--double_q", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--num_atoms", type=int, default=1)

    args = parser.parse_args()

    ray.shutdown()
    ray.init(log_to_driver=False, local_mode=True)

    main(args)
    ray.shutdown()

    send_line_notification("Lab | Training Finished")
