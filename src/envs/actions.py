from enum import IntEnum

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
                sl = env.latest_high_price if env.stop_loss else None
                env.sell(size=env.trade_size, sl=sl)

        elif action == self.BUY:
            if env.position.is_short:
                env.position.close()
            elif env.position.is_long:
                pass
            else:
                sl = env.latest_low_price if env.stop_loss else None
                env.buy(size=env.trade_size, sl=sl)
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
                sl = env.latest_high_price if env.stop_loss else None
                env.sell(size=env.trade_size, sl=sl)

        elif action == self.BUY:
            if env.position.is_short:
                env.position.close()
            elif env.position.is_long:
                pass
            else:
                sl = env.latest_low_price if env.stop_loss else None
                env.buy(size=env.trade_size, sl=sl)

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
            sl = env.latest_high_price if env.stop_loss else None
            if env.position.is_long:
                env.position.close()
                env.sell(size=env.trade_size, sl=sl)
            elif env.position.size == 0:
                env.sell(size=env.trade_size, sl=sl)

        elif action == self.LONG:
            sl = env.latest_low_price if env.stop_loss else None
            if env.position.is_short:
                env.position.close()
                env.buy(size=env.trade_size, sl=sl)
            elif env.position.size == 0:
                env.buy(size=env.trade_size, sl=sl)
        else:
            raise ValueError


class LongNeutralShort(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

    @classmethod
    def perform(self, env: "TradingEnv", action: int):
        if action == self.SHORT:
            sl = env.latest_high_price * (1 + 0.01) if env.stop_loss else None
            if env.position.is_long:
                env.position.close()
                env.sell(size=env.trade_size, sl=sl)
            elif env.position.size == 0:
                env.sell(size=env.trade_size, sl=sl)

        elif action == self.NEUTRAL:
            env.position.close()

        elif action == self.LONG:
            sl = env.latest_low_price * (1 - 0.01) if env.stop_loss else None
            if env.position.is_short:
                env.position.close()
                env.buy(size=env.trade_size, sl=sl)
            elif env.position.size == 0:
                env.buy(size=env.trade_size, sl=sl)

        else:
            raise ValueError
