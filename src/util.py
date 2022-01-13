import pathlib
from typing import Any, Dict

from ray.rllib.agents import a3c, ddpg, dqn, ppo, sac

from src.data_loader import DataLoader
from src.envs.actions import BuySell, LongNeutralShort
from src.envs.reward_func import equity_log_return_reward, initial_equity_return_reward
from src.preprocessor import Preprocessor

REWARD_FUNC = {
    "equity_log_return_reward": equity_log_return_reward,
    "initial_equity_return_reward": initial_equity_return_reward,
}
ACTIONS = {
    "BuySell": BuySell,
    "LongNeutralShort": LongNeutralShort,
}


def get_agent_class(algo: str, _config: Dict[str, Any]):

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
        config["optimization"]["actor_learning_rate"] = _config["lr"]
        config["optimization"]["critic_learning_rate"] = _config["lr"]
        config["optimization"]["entropy_learning_rate"] = _config["lr"]
        config["rollout_fragment_length"] = 4
        config["target_network_update_freq"] = 256
        config["tau"] = 1e-3
        config["Q_model"] = {
            "custom_model": _config["model"]["custom_model"],
            "fcnet_hiddens": _config["model"]["fcnet_hiddens"],
        }
        config["policy_model"] = {
            "custom_model": _config["model"]["custom_model"],
            "fcnet_hiddens": _config["model"]["fcnet_hiddens"],
        }
        _config.pop("model")

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
        config["env_config"]["reward_func"] = REWARD_FUNC[
            config["env_config"]["reward_func"]
        ]
    if isinstance(config["env_config"]["actions"], str):
        config["env_config"]["actions"] = ACTIONS[config["env_config"]["actions"]]

    config["env_config"] = {
        k: v for k, v in config["env_config"].items() if v is not None
    }
    config["evaluation_config"]["env_config"] = config["env_config"].copy()
    config["_env_test_config"] = config["env_config"].copy()

    # Divide the data according to the index
    data, features = DataLoader.prepare_data(ticker, pathlib.Path(logdir).parent.parent)
    (
        data_train,
        features_train,
        data_eval,
        features_eval,
        data_test,
        features_test,
    ) = Preprocessor.create_cv_from_index(
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

    agent_class, algo_config = get_agent_class(algo, config)
    algo_config.update(config)
    return agent_class, algo_config
