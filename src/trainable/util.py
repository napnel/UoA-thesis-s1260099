from ctypes import Union
from enum import IntEnum
import pathlib
from typing import Callable, Dict, Any

from ray import tune
from ray.rllib.agents import dqn, a3c, ppo, sac, ddpg
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
    algo = config.pop("_algo")

    # Environment Config
    window_size = config.pop("_window_size", None)
    fee = config.pop("_fee", None)
    reward_func: Union[Callable, str] = config.pop("_reward_func", None)
    actions: Union[IntEnum, str] = config.pop("_actions", None)
    stop_loss = config.pop("_stop_loss", None)
    
    # json can't save the class raw.
    if isinstance(reward_func, str): 
        reward_func = REWARD_FUNC[reward_func]
    if isinstance(actions, str):
        actions = ACTIONS[actions]

    # Data and Splits Config
    ticker = config.pop("_ticker")
    train_start = config.pop("_train_start")
    train_years = config.pop("_train_years")
    eval_years = config.pop("_eval_years")

    index = config.pop("__trial_index__")

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
