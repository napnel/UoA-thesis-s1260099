import os
import argparse
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList

from envs.trading_env import TradingEnv
from utils.data_loader import DataLoader
from utils.preprocessor import Preprocessor
from utils.backtest import backtest
from utils.callbacks import CustomEvalCallback
from utils.utils import set_random_seed, send_line_notification, ALGOS


def train(algo: BaseAlgorithm, n_timesteps: int, n_cpus: int, verbose: int):
    # data = DataLoader.fetch_data("BTC-USD", interval='1h')
    data = pd.read_csv("./data/LTCUSD.csv", index_col="Date").rename(index=pd.to_datetime)
    data_len = len(data)

    data_train = data.iloc[: int(data_len * 0.75), :]
    data_test = data.iloc[int(data_len * 0.75) :, :]
    features_train = Preprocessor.extract_features(data_train)
    features_test = Preprocessor.extract_features(data_test)
    data_train, features_train = Preprocessor.align_date(data_train, features_train)
    data_test, features_test = Preprocessor.align_date(data_test, features_test)
    print(f"Train Sapn: {data_train.index[0]} to {data_train.index[-1]}")
    print(f"Test: Span {data_test.index[0]} to {data_test.index[-1]}")
    n_cpus = 6

    scaler = StandardScaler()
    features_train = pd.DataFrame(scaler.fit_transform(features_train), index=data_train.index)
    features_test = pd.DataFrame(scaler.transform(features_test), index=data_test.index)

    vec_env_train = make_vec_env(TradingEnv, n_envs=n_cpus, env_kwargs={"df": data_train, "features": features_train}, vec_env_cls=SubprocVecEnv)
    vec_env_eval = make_vec_env(TradingEnv, n_envs=1, env_kwargs={"df": data_test, "features": features_test}, vec_env_cls=SubprocVecEnv)

    algo_name = algo.__name__
    model: BaseAlgorithm = algo("MlpPolicy", vec_env_train, device="cpu")
    logger = configure_logger(verbose, tensorboard_log="./logs/", tb_log_name=algo_name)
    model.set_logger(logger)

    # Make Callback
    eval_callback = CustomEvalCallback(vec_env_eval, best_model_save_path=model.logger.dir, eval_freq=n_timesteps / 10 / n_cpus, n_eval_episodes=1)
    callback = CallbackList([eval_callback])

    # Training
    start_time = time.time()
    model = model.learn(n_timesteps, callback=callback)
    # model = model.learn(n_timesteps, eval_env=vec_env_eval, n_eval_episodes=1, eval_freq=n_timesteps / 10 / n_cpus)
    total_time = time.time() - start_time
    print(f"Took {total_time:.2f}s, {n_timesteps / total_time:.2f} FPS")

    # Backtest
    model = algo.load(os.path.join(model.logger.dir, "best_model"))
    stats = pd.DataFrame()
    stats["train"] = backtest(model, TradingEnv(data_train, features_train), plot=False)
    stats["test"] = backtest(model, TradingEnv(data_test, features_test), plot=True, plot_filename=os.path.join(model.logger.dir, "backtest"))
    stats.to_csv(os.path.join(model.logger.dir, "backtest_stats.csv"))
    print(stats)


if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--n_timesteps", help="Overwrite the number of timesteps", default=3e5, type=int)
    parser.add_argument("--data_path", help="Data input to the agent", default="", type=str)
    parser.add_argument("--n_cpus", help="The number of cpus for multiprocessing", default="8", type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    # parser.add_argument(
    #     "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    # )
    args = parser.parse_args()
    set_random_seed(args.seed)
    train(
        algo=ALGOS[args.algo],
        n_timesteps=args.n_timesteps,
        n_cpus=args.n_cpus,
        verbose=args.verbose,
    )
    send_line_notification("Training | Finished")
