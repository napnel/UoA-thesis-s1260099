from enum import IntEnum
import gym
from gym import spaces
from src.envs.core.dummy_environment import TradingEnv


class BuySell(IntEnum):
    SELL = 0
    BUY = 1
    #   Neutral = 2
    @classmethod
    def perform(self, env: "TradingEnv", action: int):
        if action == self.SELL:
            if env.position.is_long:
                env.position.close()
            elif env.position.is_short:
                pass
            else:
                env.sell(size=env.trade_size, sl=env.sl_price)

        elif action == self.BUY:
            if env.position.is_short:
                env.position.close()
            elif env.position.is_long:
                pass
            else:
                env.buy(size=env.trade_size, sl=env.sl_price)
        else:
            raise ValueError


class BuyHoldSell(IntEnum):
    NEUTRAL = 0
    SELL = 1
    BUY = 2

    @classmethod
    def perform(self, env: "TradingEnv", action: int):
        if action == self.SELL:
            if env.position.is_long:
                env.position.close()
            elif env.position.is_short:
                pass
            else:
                env.sell(size=env.trade_size, sl=env.sl_price)

        elif action == self.BUY:
            if env.position.is_short:
                env.position.close()
            elif env.position.is_long:
                pass
            else:
                env.buy(size=env.trade_size, sl=env.sl_price)

        elif action == self.NEUTRAL:
            pass
        else:
            raise ValueError


class LonglShort(IntEnum):
    SHORT = 0
    LONG = 1

    @classmethod
    def perform(self, env: "TradingEnv", action: int):
        if action == self.SHORT:
            if env.position.is_long:
                env.position.close()
                env.sell(size=env.trade_size, sl=env.sl_price)
            elif env.position.size == 0:
                env.sell(size=env.trade_size, sl=env.sl_price)

        elif action == self.LONG:
            if env.position.is_short:
                env.position.close()
                env.buy(size=env.trade_size, sl=env.sl_price)
            elif env.position.size == 0:
                env.buy(size=env.trade_size, sl=env.sl_price)
        else:
            raise ValueError


class LongNeutralShort(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

    @classmethod
    def perform(self, env: "TradingEnv", action: int):
        if action == self.SHORT:
            if env.position.is_long:
                env.position.close()
                env.sell(size=env.trade_size, sl=env.sl_price)
            elif env.position.size == 0:
                env.sell(size=env.trade_size, sl=env.sl_price)

        elif action == self.NEUTRAL:
            env.position.close()

        elif action == self.LONG:
            if env.position.is_short:
                env.position.close()
                env.buy(size=env.trade_size, sl=env.sl_price)
            elif env.position.size == 0:
                env.buy(size=env.trade_size, sl=env.sl_price)
        else:
            raise ValueError
