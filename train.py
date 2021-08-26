import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from envs.trading_env import TradingEnv
from utils.data_loader import DataLoader
from utils.preprocessor import Preprocessor
from utils.backtest import backtest
from utils.utils import reduce_mem_usage, fixed_seed


def main():
    # data = DataLoader.fetch_data("BTC-USD", interval='1h')
    data = pd.read_csv("./data/LTCUSD.csv", index_col="Date").rename(index=pd.to_datetime)
    data[data["Volume"] == 0] = np.nan
    data = data.fillna(method="ffill").fillna(method="bfill")
    data_len = len(data)

    data_train = data.iloc[: int(data_len * 0.75), :]
    data_test = data.iloc[int(data_len * 0.75) :, :]
    features_train = Preprocessor.extract_features(data_train)
    features_test = Preprocessor.extract_features(data_test)
    data_train, features_train = Preprocessor.align_date(data_train, features_train)
    data_test, features_test = Preprocessor.align_date(data_test, features_test)
    print(f"Train Sapn: {data_train.index[0]} to {data_train.index[-1]}")
    print(f"Test: Span {data_test.index[0]} to {data_test.index[-1]}")
    n_cpus = 8

    # transformer = PowerTransformer()
    # features_train = transformer.fit_transform(features_train)
    # features_test = transformer.transform(features_test)
    scaler = StandardScaler()
    features_train = pd.DataFrame(scaler.fit_transform(features_train), index=data_train.index)
    features_test = pd.DataFrame(scaler.transform(features_test), index=data_test.index)

    env_train = make_vec_env(TradingEnv, n_envs=n_cpus, env_kwargs={"df": data_train, "features": features_train}, vec_env_cls=SubprocVecEnv)
    env_valid = TradingEnv(data_test, features_test)
    model = PPO("MlpPolicy", env_train, device="cpu", tensorboard_log="./logs/")

    total_timesteps = 15e4
    eval_freq = total_timesteps / 10 / n_cpus

    start_time = time.time()
    model = model.learn(total_timesteps, eval_env=env_valid, n_eval_episodes=1, eval_freq=eval_freq)
    total_time = time.time() - start_time
    print(f"Took {total_time:.2f}s, {total_timesteps / total_time:.2f} FPS")

    stats = pd.DataFrame()
    stats["train"] = backtest(model, TradingEnv(data_train, features_train), plot=False)
    stats["test"] = backtest(model, env_valid, plot=True)
    print(stats)

    model.save(f"./models/PPO")
    with open(f"./models/env_valid.pickle", "wb") as f:
        pickle.dump(env_valid, f)


if __name__ == "__main__":
    fixed_seed()
    main()
