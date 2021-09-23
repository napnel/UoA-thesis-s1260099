import torch.nn as nn
from stable_baselines3 import PPO, A2C, DQN

ALGOS = {
    "a2c": A2C,
    "dqn": DQN,
    "ppo": PPO,
}

NETWORKS = {
    "mlp": "MlpPolicy",
    "cnn": "CnnPolicy",
}

PPO_HYPERPARAMETER = {}
A2C_HYPERPARAMETER = {
    "normalize_advantage": True,
}

ENV_KWARGS = {
    "window_size": 20,
    "fee": 0.001,  # [%]
}


POLICY_KWARGS = {
    "activation_fn": nn.ReLU,
    "net_arch": [64, dict(pi=[32, 16], vf=[32, 16])],
}

TRAINER_CONFIG = {
    # "env": "DescTradingEnv",
    # "env_config": {
    #     "df": data_train,
    #     "features": features_train,
    #     "reward_func": reward_func,
    #     "window_size": window_size,
    # },
    # "evaluation_num_workers": 1,
    # "evaluation_interval": 1,
    # "evaluation_num_episodes": 1,
    # "evaluation_config": {
    #     "env_config": {
    #         "df": data_eval,
    #         "features": features_eval,
    #         "reward_func": reward_func,
    #         "window_size": window_size,
    #     },
    #     "explore": False,
    # },
    "model": {
        # By default, the MODEL_DEFAULTS dict above will be used.
        # Change individual keys in that dict by overriding them, e.g.
        "fcnet_hiddens": [128, 64],
    },
    # "vf_loss_coeff": 0.5,
    # "entropy_coeff": 0.01,
    "num_workers": 4,  # parallelism
    "framework": "torch",
    "log_level": "WARN",  # "WARN", "DEBUG"
    "seed": 0,
    "observation_filter": "MeanStdFilter",
}
