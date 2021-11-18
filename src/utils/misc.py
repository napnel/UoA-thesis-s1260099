import os
import random
import requests
import pathlib
from typing import Any, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def set_random_seed(seed=0):
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def send_line_notification(message):
    """
    If you have line notify token and make .env file in utils folder, send message to LINE
    """
    load_dotenv(verbose=True)
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)
    line_notify_token = os.environ.get("LINE_NOTIFY_TOKEN")
    endpoint = "https://notify-api.line.me/api/notify"
    payload = {"message": f"\n{message}"}
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    requests.post(endpoint, data=payload, headers=headers)


def visualize_network(agent, device="cpu", policy_id="default_policy"):
    import torch
    from torchviz import make_dot
    from ray.rllib.execution.rollout_ops import ParallelRollouts

    rollouts = ParallelRollouts(agent.workers, mode="async")
    batch = next(rollouts)
    torch_batch = batch.to_device("cpu")

    model = agent.get_policy(policy_id).model
    y = model(torch_batch)[0]
    return make_dot(y, params=dict(model.named_parameters()))


def get_agent_class(algo: str):
    from ray.rllib.agents import dqn, a3c, ppo, sac, ddpg

    if algo == "DQN":
        agent = dqn.DQNTrainer
        config = dqn.DEFAULT_CONFIG.copy()

    elif algo == "A2C":
        agent = a3c.A2CTrainer
        config = a3c.DEFAULT_CONFIG.copy()

    elif algo == "PPO":
        agent = ppo.PPOTrainer
        config = ppo.DEFAULT_CONFIG.copy()

    elif algo == "SAC":
        agent = sac.SACTrainer
        config = sac.DEFAULT_CONFIG.copy()

    elif algo == "DDPG":
        agent = ddpg.DDPGTrainer
        config = ddpg.DEFAULT_CONFIG.copy()

    else:
        raise ValueError

    return agent, config


def prepare_config_for_agent(config: Dict[str, Any], logdir: str):
    from ray import tune
    from src.utils import DataLoader, Preprocessor

    algo = config.pop("_algo")

    # Environment Config
    window_size = config.pop("_window_size", None)
    fee = config.pop("_fee", None)
    reward_func = config.pop("_reward_func", None)
    actions = config.pop("_actions", None)
    stop_loss = config.pop("_stop_loss", None)

    # Data and Splits Config
    ticker = config.pop("_ticker")
    train_start = config.pop("_train_start")
    train_years = config.pop("_train_years")
    eval_years = config.pop("_eval_years")
    index = config.pop(tune.suggest.repeater.TRIAL_INDEX)

    # Divide the data according to the index
    data, features = DataLoader.prepare_data(ticker, pathlib.Path(logdir).parent)
    data_train, features_train, data_eval, features_eval = Preprocessor.create_cv_from_index(
        data,
        features,
        index,
        train_years,
        eval_years,
        train_start,
    )

    config["env_config"] = {
        "data": data_train,
        "features": features_train,
        "window_size": window_size,
        "fee": fee,
        "reward_func": reward_func,
        "actions": actions,
        "stop_loss": stop_loss,
    }
    config["evaluation_config"]["env_config"] = {
        "data": data_eval,
        "features": features_eval,
        "window_size": window_size,
        "fee": fee,
        "reward_func": reward_func,
        "actions": actions,
        "stop_loss": stop_loss,
    }
    config["env_config"] = {k: v for k, v in config["env_config"].items() if v is not None}
    config["evaluation_config"]["env_config"] = {k: v for k, v in config["evaluation_config"]["env_config"].items() if v is not None}

    agent_class, algo_config = get_agent_class(algo)
    algo_config.update(config)
    return agent_class, algo_config


def get_env(env_name: str, env_config={}):
    if env_name == "DescTradingEnv":
        from envs.trading_env import DescTradingEnv

        env = DescTradingEnv(**env_config)

    elif env_name == "ContTradingEnv":
        from envs.trading_env import ContTradingEnv

        env = ContTradingEnv(**env_config)

    else:
        raise ValueError

    return env


def clean_result(result: Dict[str, Any]):
    output = {}
    output["episode_reward_max"] = result["episode_reward_max"]
    output["episode_reward_mean"] = result["episode_reward_mean"]
    output["episode_reward_min"] = result["episode_reward_min"]
    output["evaluation"] = {}
    output["evaluation"]["episode_reward_max"] = result["evaluation"]["episode_reward_max"]
    output["evaluation"]["episode_reward_mean"] = result["evaluation"]["episode_reward_mean"]
    output["evaluation"]["episode_reward_min"] = result["evaluation"]["episode_reward_min"]
    output["timesteps"] = result["timesteps_total"]
    output["iteration"] = result["training_iteration"]
    return output


def clean_stats(stats: pd.Series):
    output = stats.loc[
        [
            "Start",
            "End",
            "Return [%]",
            "Buy & Hold Return [%]",
            "Return (Ann.) [%]",
            "Volatility (Ann.) [%]",
            "Sharpe Ratio",
            "Max. Drawdown [%]",
            "Max. Drawdown Duration",
        ]
    ]
    return output
