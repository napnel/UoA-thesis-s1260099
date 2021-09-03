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

POLICY_KWARGS = {
    "activation_fn": nn.PReLU,
    "net_arch": [32, dict(pi=[16, 8], vf=[16, 8])],
}
