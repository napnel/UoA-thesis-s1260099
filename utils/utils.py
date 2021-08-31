import os
import random
import requests

import numpy as np
import pandas as pd
import torch
import gym
from gym import spaces
from dotenv import load_dotenv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import obs_as_tensor


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


def get_action_prob(model: BaseAlgorithm, env: gym.Env, obaservation: spaces.Box):
    obs = obaservation.reshape((-1,) + env.observation_space.shape)
    obs = obs_as_tensor(obs, model.device)
    latent_pi, _, latent_sde = model.policy._get_latent(obs)
    distribution = model.policy._get_action_dist_from_latent(latent_pi, latent_sde)
    action_prob = distribution.distribution.probs
    return action_prob.detach().numpy()[0]
