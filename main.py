import os
import glob
import shutil
import argparse

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

from src.envs import BaseTradingEnv
from src.envs.reward_func import equity_log_return_reward
from src.utils import DataLoader, Preprocessor, backtest
from src.utils.logger import custom_log_creator, create_log_filename
from src.utils.misc import clean_result, clean_stats, send_line_notification, get_agent_class

TICKERS = ["ES=F", "^GSPC", "^N225"]


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

    data_train, features_train, data_eval, features_eval = Preprocessor.time_train_test_split(
        data,
        features,
        train_start="2009-01-01",
        train_end="2018-01-01",
        eval_start="2018-01-01",
        eval_end="2021-01-01",
        scaling=True,
    )

    reward_func = equity_log_return_reward
    user_config = {
        "env": "BaseTradingEnv",
        "env_config": {
            "data": data_train,
            "features": features_train,
            "reward_func": reward_func,
            "window_size": args.window_size,
        },
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "env_config": {
                "data": data_eval,
                "features": features_eval,
                "reward_func": reward_func,
                "window_size": args.window_size,
            },
            "explore": False,
        },
        "num_workers": 2,
        "framework": "torch",
        "log_level": "WARN",
        "timesteps_per_iteration": 5000,
        "num_gpus": 0,
        "seed": 3407,
    }

    agent_class, config = get_agent_class(args.algo)
    config.update(user_config)
    # if args.algo == "DQN":
    #     config["hiddens"] = [64, 16]

    # if args.algo == "Rainbow":
    #     config["hiddens"] = [64, 16]
    #     config["noisy"] = True
    #     config["n_step"] = 10
    #     config["num_atoms"] = 51

    # if args.algo == "SAC":
    #     config["Q_model"] = {"fcnet_hiddens": [256, 64]}
    #     config["policy_model"] = {"fcnet_hiddens": [256, 64]}

    # log_filename = create_log_filename(args)
    log_filename = args.algo

    agent = agent_class(config=config, logger_creator=custom_log_creator(args.logdir, log_filename))
    print(agent.get_policy().model)

    max_episode_reward_mean = -1
    while True:
        result = agent.train()
        print(pretty_print(result))
        # print(pretty_print(clean_result(result)))
        if result["evaluation"]["episode_reward_mean"] > max_episode_reward_mean:
            max_episode_reward_mean = result["evaluation"]["episode_reward_mean"]
            best_checkpoint = agent.save(os.path.join(agent.logdir, "best"))

        if result["timesteps_total"] >= args.stop_timesteps:
            break

    last_checkpoint = agent.save()

    env_train = BaseTradingEnv(**config["env_config"])
    env_eval = BaseTradingEnv(**config["evaluation_config"]["env_config"])

    backtest(env_train, agent, save_dir=os.path.join(agent.logdir, "last-stats-train"), plot=True)
    backtest(env_eval, agent, save_dir=os.path.join(agent.logdir, "last-stats-eval"), plot=True)

    agent.restore(best_checkpoint)
    backtest(env_eval, agent, save_dir=os.path.join(agent.logdir, "best-stats-eval"), plot=True)
    backtest(env_eval, "Buy&Hold", save_dir=os.path.join(agent.logdir, "buy&hold-stats-eval"), plot=True)
    print("last checkpoint saved at", last_checkpoint)
    print("best checkpoint saved at", best_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["DQN", "Rainbow", "A2C", "PPO", "SAC"])
    parser.add_argument("--window_size", type=int, default=9)
    parser.add_argument("--hiddens", type=int, nargs="*", default=None)
    parser.add_argument("--ticker", type=str, default="^N225")
    parser.add_argument("--stop-timesteps", type=int, default=int(1e5))
    parser.add_argument("--logdir", type=str, default="./ray_results")

    args = parser.parse_args()

    ray.shutdown()
    ray.init(log_to_driver=False)

    main(args)
    ray.shutdown()

    send_line_notification("Lab | Training Finished")
