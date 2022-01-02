from ctypes import Union
from enum import IntEnum
import pathlib
from typing import Callable, Dict, Any

from ray import tune
from ray.rllib.agents import dqn, a3c, ppo, sac, ddpg
from sklearn import feature_selection
from src.utils import DataLoader, Preprocessor
from src.envs.actions import BuySell, LongNeutralShort
from src.envs.reward_func import equity_log_return_reward, initial_equity_return_reward

REWARD_FUNC = {
    "equity_log_return_reward": equity_log_return_reward,
    "initial_equity_return_reward": initial_equity_return_reward,
}
ACTIONS = {
    "BuySell": BuySell,
    "LongNeutralShort": LongNeutralShort,
}


def get_agent_class(algo: str):

    if algo == "DQN":
        agent = dqn.DQNTrainer
        config = dqn.DEFAULT_CONFIG.copy()

    elif algo == "A2C":
        from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG

        agent = a3c.A2CTrainer
        config = A2C_DEFAULT_CONFIG.copy()
        config["min_iter_time_s"] = 0

    elif algo == "A3C":
        agent = a3c.A3CTrainer
        config = a3c.DEFAULT_CONFIG.copy()
        config["min_iter_time_s"] = 0

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


def prepare_config_for_agent(_config: Dict[str, Any], logdir: str):
    config = _config.copy()
    algo = config.pop("_algo")
    ticker = config.pop("_ticker")
    cv_config = config.pop("_cv_config")
    index = config.pop("__trial_index__")

    if isinstance(config["env_config"]["reward_func"], str):
        config["env_config"]["reward_func"] = REWARD_FUNC[config["env_config"]["reward_func"]]
    if isinstance(config["env_config"]["actions"], str):
        config["env_config"]["actions"] = ACTIONS[config["env_config"]["actions"]]

    config["env_config"] = {k: v for k, v in config["env_config"].items() if v is not None}
    config["evaluation_config"]["env_config"] = config["env_config"].copy()
    config["_env_test_config"] = config["env_config"].copy()

    # Divide the data according to the index
    data, features = DataLoader.prepare_data(ticker, pathlib.Path(logdir).parent)
    data_train, features_train, data_eval, features_eval, data_test, features_test = Preprocessor.create_cv_from_index(
        data,
        features,
        index,
        cv_config["train_years"],
        cv_config["eval_years"],
        cv_config["train_start"],
    )

    config["env_config"]["data"] = data_train
    config["env_config"]["features"] = features_train

    config["evaluation_config"]["env_config"]["data"] = data_eval
    config["evaluation_config"]["env_config"]["features"] = features_eval

    config["_env_test_config"]["data"] = data_test
    config["_env_test_config"]["features"] = features_test

    agent_class, algo_config = get_agent_class(algo)
    algo_config.update(config)
    return agent_class, algo_config
