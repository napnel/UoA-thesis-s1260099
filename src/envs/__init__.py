from ray.tune.registry import register_env
from src.envs.environment import TradingEnv

register_env("TradingEnv", lambda env_config: TradingEnv(**env_config))
