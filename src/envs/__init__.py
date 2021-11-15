from ray.tune.registry import register_env
from src.envs.base_env import BaseTradingEnv

# from src.envs.trading_env import DescTradingEnv, ContTradingEnv, MarketMakingEnv

register_env("BaseTradingEnv", lambda env_config: BaseTradingEnv(**env_config))
# register_env("DescTradingEnv", lambda env_config: DescTradingEnv(**env_config))
# register_env("ContTradingEnv", lambda env_config: ContTradingEnv(**env_config))
# register_env("MarketMakingEnv", lambda env_config: MarketMakingEnv(**env_config))
