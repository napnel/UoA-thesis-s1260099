import os
import glob
import shutil
import argparse

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

from src.envs.trading_env import DescTradingEnv
from src.envs.reward_func import equity_log_return_reward
from src.models.batch_norm import TorchBatchNormModel
from src.utils import DataLoader, Preprocessor, backtest
from src.utils.logger import custom_log_creator
from src.utils.misc import clean_result, clean_stats, send_line_notification


def main(args):
    tickers = ["ES=F", "^GSPC", "^N225"]
    data = DataLoader.fetch_data(args.ticker, interval="1d")
    data_train, features_train, data_eval, features_eval = Preprocessor.preprocessing(
        data=data,
        train_start="2009-01-01",
        train_end="2019-01-01",
        eval_start="2019-01-01",
        eval_end="2021-01-01",
    )

    reward_func = equity_log_return_reward
    config = {
        "env": "DescTradingEnv",
        "env_config": {
            "df": data_train,
            "features": features_train,
            "reward_func": reward_func,
            "window_size": args.window_size,
            "fee": args.fee,
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
                "fee": args.fee,
            },
            "explore": False,
            # "explore": True,
        },
        "model": {
            # By default, the MODEL_DEFAULTS dict above will be used.
            # Change individual keys in that dict by overriding them, e.g.
            "fcnet_hiddens": args.hiddens,
            "custom_model": args.model,
            "vf_share_layers": False,
        },
        "num_workers": 4,  # parallelism
        "framework": "torch",
        "log_level": "WARN",  # "WARN", "DEBUG"
        "seed": 3407,
    }

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)
    agent = ppo.PPOTrainer(config=ppo_config, logger_creator=custom_log_creator("./ray_results", args.log_file))

    if args.checkpoint:
        agent.restore(args.checkpoint)

    if args.backtest:
        env_eval = DescTradingEnv(**config["evaluation_config"]["env_config"])
        stats = backtest(env_eval, agent, plot=True, plot_filename=os.path.join(agent.logdir, "backtest"))
        print(clean_stats(stats))

    else:
        while True:
            result = agent.train()
            print(pretty_print(clean_result(result)))
            if result["timesteps_total"] >= args.stop_timesteps:
                break

        checkpoint = agent.save()

        env_train = DescTradingEnv(**config["env_config"])
        env_eval = DescTradingEnv(**config["evaluation_config"]["env_config"])
        backtest(env_train, agent, plot=True, plot_filename=os.path.join(agent.logdir, "backtest_train"))
        stats = backtest(env_eval, agent, plot=True, plot_filename=os.path.join(agent.logdir, "backtest_eval"))
        stats.to_csv(f"{agent.logdir}/stats.csv", header=False)
        print(clean_stats(stats))
        print("checkpoint saved at", checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ticker", type=str, default="^N225")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--hiddens", nargs="*", default=[128, 128])
    parser.add_argument("--stop-timesteps", type=int, default=1e6)
    parser.add_argument("--log-file", type=str, default="PPO")

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    # remove folder has no checkpoints
    ray_results = glob.glob("./ray_results/*")
    for folder in ray_results:
        checkpoints = glob.glob(f"{folder}/checkpoint*")
        if len(checkpoints) == 0:
            print(f"Remove {folder}")
            shutil.rmtree(folder)

    ray.shutdown()
    ray.init(log_to_driver=False)
    ModelCatalog.register_custom_model("bn_model", TorchBatchNormModel)

    main(args)
    ray.shutdown()

    send_line_notification("Lab | Training Finished")
