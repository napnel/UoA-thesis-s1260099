from ray.tune.registry import register_env

from envs.trading_env import DescTradingEnv, ContTradingEnv

register_env("DescTradingEnv", lambda env_config: DescTradingEnv(**env_config))
register_env("ContTradingEnv", lambda env_config: ContTradingEnv(**env_config))
