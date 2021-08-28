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
