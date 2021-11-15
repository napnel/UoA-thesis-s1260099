from argparse import Action
import gym
from gym import spaces
import numpy as np
import pandas as pd
import sys
from math import copysign
from enum import Enum
from copy import copy
import warnings
from typing import Optional, Dict, Callable, List, Union

from src.envs.reward_func import equity_log_return_reward


class BS(Enum):
    Sell = 0
    Buy = 1
    # Neutral = 2
    def perform(env: "BaseTradingEnv", action):
        if action == BS.Sell.value:
            if env.position.is_long:
                env.position.close()
            elif env.position.is_short:
                pass
            else:
                env.sell(size=env.trade_size, sl=env.sl_price)

        elif action == BS.Buy.value:
            if env.position.is_short:
                env.position.close()
            elif env.position.is_long:
                pass
            else:
                env.buy(size=env.trade_size, sl=env.sl_price)
        else:
            raise ValueError


class LNS(Enum):
    Short = -1
    Neutral = 0
    Long = 1

    def perform(env: "BaseTradingEnv", action):
        action = action - 1
        if action == LNS.Short.value:
            if env.position.is_long:
                env.position.close()
                env.sell(size=env.trade_size, sl=env.sl_price)
            elif env.position.size == 0:
                env.sell(size=env.trade_size, sl=env.sl_price)

        elif action == LNS.Neutral.value:
            env.position.close()

        elif action == LNS.Long.value:
            if env.position.is_short:
                env.position.close()
                env.buy(size=env.trade_size, sl=env.sl_price)
            elif env.position.size == 0:
                env.buy(size=env.trade_size, sl=env.sl_price)
        else:
            raise ValueError


class Position:
    def __init__(self, env: "BaseTradingEnv"):
        self.__env = env

    def __bool__(self):
        return self.size != 0

    @property
    def size(self) -> float:
        return sum(trade.size for trade in self.__env.trades)

    @property
    def pnl(self) -> float:
        return sum(trade.pnl for trade in self.__env.trades)

    @property
    def pnl_pct(self) -> float:
        weights = np.abs([trade.size for trade in self.__env.trades])
        weights = weights / weights.sum()
        pnl_pcts = np.array([trade.pnl_pct for trade in self.__env.trades])
        return (pnl_pcts * weights).sum()

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    def close(self, portion: float = 1.0):
        for trade in self.__env.trades:
            trade.close(portion)

    def __repr__(self):
        return f"<Position: {self.size} ({len(self.__env.trades)} trades)>"


class Order:
    def __init__(
        self,
        broker: "BaseTradingEnv",
        size: float,
        limit_price: float = None,
        stop_price: float = None,
        sl_price: float = None,
        tp_price: float = None,
        parent_trade: "Trade" = None,
    ):
        self.__broker = broker
        assert size != 0
        self.__size = size
        self.__limit_price = limit_price
        self.__stop_price = stop_price
        self.__sl_price = sl_price
        self.__tp_price = tp_price
        self.__parent_trade = parent_trade

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f"_{self.__class__.__qualname__}__{k}", v)
        return self

    def __repr__(self):
        return "<Order {}>".format(
            ", ".join(
                f"{param}={round(value, 5)}"
                for param, value in (
                    ("size", self.__size),
                    ("limit", self.__limit_price),
                    ("stop", self.__stop_price),
                    ("sl", self.__sl_price),
                    ("tp", self.__tp_price),
                    ("contingent", self.is_contingent),
                )
                if value is not None
            )
        )

    def cancel(self):
        """Cancel the order."""
        self.__broker.orders.remove(self)
        trade = self.__parent_trade
        if trade:
            if self is trade._sl_order:
                trade._replace(sl_order=None)
            elif self is trade._tp_order:
                trade._replace(tp_order=None)
            else:
                # XXX: https://github.com/kernc/backtesting.py/issues/251#issuecomment-835634984 ???
                assert False

    # Fields getters

    @property
    def size(self) -> float:
        return self.__size

    @property
    def limit(self) -> Optional[float]:
        return self.__limit_price

    @property
    def stop(self) -> Optional[float]:
        return self.__stop_price

    @property
    def sl(self) -> Optional[float]:
        return self.__sl_price

    @property
    def tp(self) -> Optional[float]:
        return self.__tp_price

    @property
    def parent_trade(self):
        return self.__parent_trade

    # Extra properties

    @property
    def is_long(self):
        return self.__size > 0

    @property
    def is_short(self):
        return self.__size < 0

    @property
    def is_contingent(self):
        return bool(self.__parent_trade)


class Trade:
    def __init__(self, broker: "BaseTradingEnv", size: int, entry_price: float, entry_bar):
        self.__env = broker
        self.__size = size
        self.__entry_price = entry_price
        self.__exit_price: Optional[float] = None
        self.__entry_bar: int = entry_bar
        self.__exit_bar: Optional[int] = None
        self.__sl_order: Optional[Order] = None
        self.__tp_order: Optional[Order] = None

    def __repr__(self):
        return (
            f'<Trade size={self.__size} time={self.__entry_bar}-{self.__exit_bar or ""} '
            f'price={self.__entry_price}-{self.__exit_price or ""} pnl={self.pnl:.0f}>'
        )

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f"_{self.__class__.__qualname__}__{k}", v)
        return self

    def _copy(self, **kwargs):
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1.0):
        """Place new `Order` to close `portion` of the trade at next market price."""
        assert 0 < portion <= 1, "portion must be a fraction between 0 and 1"
        size = copysign(max(1, round(abs(self.__size) * portion)), -self.__size)
        order = Order(self.__env, size, parent_trade=self)
        self.__env.orders.insert(0, order)

    # Fields getters

    @property
    def size(self):
        return self.__size

    @property
    def entry_price(self) -> float:
        return self.__entry_price

    @property
    def exit_price(self) -> Optional[float]:
        return self.__exit_price

    @property
    def entry_bar(self) -> int:
        return self.__entry_bar

    @property
    def exit_bar(self) -> Optional[int]:
        return self.__exit_bar

    @property
    def _sl_order(self):
        return self.__sl_order

    @property
    def _tp_order(self):
        return self.__tp_order

    # Extra properties

    @property
    def entry_time(self) -> Union[pd.Timestamp, int]:
        return self.__env._data.index[self.__entry_bar]

    @property
    def exit_time(self) -> Optional[Union[pd.Timestamp, int]]:
        if self.__exit_bar is None:
            return None
        return self.__env._data.index[self.__exit_bar]

    @property
    def is_long(self):
        return self.__size > 0

    @property
    def is_short(self):
        return not self.is_long

    @property
    def pnl(self):
        price = self.__exit_price or self.__env.closing_price
        return self.__size * (price - self.__entry_price)

    @property
    def pnl_pct(self):
        price = self.__exit_price or self.__env.closing_price
        return copysign(1, self.__size) * (price / self.__entry_price - 1)

    @property
    def value(self):
        price = self.__exit_price or self.__env.closing_price
        return abs(self.__size) * price

    # SL/TP management API

    @property
    def sl(self):
        """
        Stop-loss price at which to close the trade.
        This variable is writable. By assigning it a new price value,
        you create or modify the existing SL order.
        By assigning it `None`, you cancel it.
        """
        return self.__sl_order and self.__sl_order.stop

    @sl.setter
    def sl(self, price: float):
        self.__set_contingent("sl", price)

    @property
    def tp(self):
        """
        Take-profit price at which to close the trade.
        This property is writable. By assigning it a new price value,
        you create or modify the existing TP order.
        By assigning it `None`, you cancel it.
        """
        return self.__tp_order and self.__tp_order.limit

    @tp.setter
    def tp(self, price: float):
        self.__set_contingent("tp", price)

    def __set_contingent(self, type, price):
        assert type in ("sl", "tp")
        assert price is None or 0 < price < np.inf
        attr = f"_{self.__class__.__qualname__}__{type}_order"
        order: Order = getattr(self, attr)
        if order:
            order.cancel()
        if price:
            kwargs = dict(stop=price) if type == "sl" else dict(limit=price)
            order = self.__env.new_order(-self.size, trade=self, **kwargs)
            setattr(self, attr, order)


class BaseTradingEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int = 5,
        fee: float = 0.001,
        reward_func: Callable = equity_log_return_reward,
        actions: Enum = BS,
        stop_loss: bool = False,
        debug: bool = False,
    ):
        """ """
        assert len(data) == len(features)
        self.data = data.copy()
        self.features = features.copy()
        self.fee = fee
        self.window_size = window_size
        self.reward_func = reward_func
        self.actions = actions
        self.stop_loss = stop_loss
        self.debug = debug

        self.current_step = 0
        self.initial_assets = 100000
        self.assets = self.initial_assets
        self.trade_size = self.assets // self.data["High"].max()
        self.position: Optional[Position] = Position(self)
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.observation_size = len(self.features.columns) + 1

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.observation_size), dtype=np.float32)

        self._leverage = 1
        self._hedging = False

    def reset(self):
        self.done = False
        self.next_done = False
        self.action = None
        self.reward = 0
        self.current_step = self.window_size
        self.assets = self.initial_assets
        self.trade_size = self.assets // self.data["High"].max()
        self.position = Position(self)
        self.orders = []
        self.trades = []
        self.equity_curve = [self.equity]
        self.closed_trades = []

        features_obs = self.features.iloc[: self.window_size, :].values
        account_obs = np.tile([self.position.pnl_pct], (self.window_size, 1))
        self.observation = np.hstack((features_obs, account_obs))
        return self.observation

    def step(self, action):
        self.action = action

        # Trade Start
        if self.next_done:
            self.done = True
            self.position.close()

        else:
            self.actions.perform(self, action)

        if self.debug:
            self.render()
        # Trade End
        self.current_step += 1
        self._process_orders()

        self.equity_curve.append(self.equity)
        if self.equity < self.closing_price:
            self.next_done = True

        self.next_done = True if self.current_step >= len(self.data) - 3 else False

        self.observation = self.next_observation
        self.reward = self.reward_func(self)
        self.info = {}

        return self.observation, self.reward, self.done, self.info

    def render(self, mode="human"):
        print("===" * 5, f"Environment ({self.current_time})", "===" * 5)
        print(f"Price: {self.closing_price}")
        print(f"Assets: {self.assets}")
        print(f"Equity: {self.equity}")
        print(f"Orders: {self.orders}")
        print(f"Trades: {self.trades}")
        print(f"Position: {self.position}")
        print(f"Closed Trades: {self.closed_trades}")
        print(f"Action: {self.action}, Reward: {self.reward}, Done: {self.done}\n")

    class __FULL_EQUITY(float):
        def __repr__(self):
            return ".9999"

    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def buy(
        self,
        size: Optional[float] = _FULL_EQUITY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        sl: float = None,
        tp: float = None,
    ):
        assert 0 < size < 1 or round(size) == size
        return self.new_order(size, limit_price, stop_price, sl, tp)

    def sell(
        self,
        size: Optional[float] = _FULL_EQUITY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        sl: float = None,
        tp: float = None,
    ):
        assert 0 < size < 1 or round(size) == size
        return self.new_order(-size, limit_price, stop_price, sl, tp)

    def new_order(self, size: float, limit: float = None, stop: float = None, sl: float = None, tp: float = None, *, trade: Trade = None):
        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        is_long = size > 0
        adjusted_price = self._adjusted_price(size)

        if is_long:
            if not (sl or -np.inf) < (limit or stop or adjusted_price) < (tp or np.inf):
                raise ValueError("Long orders require: " f"SL ({sl}) < LIMIT ({limit or stop or adjusted_price}) < TP ({tp})")
        else:
            if not (tp or -np.inf) < (limit or stop or adjusted_price) < (sl or np.inf):
                raise ValueError("Short orders require: " f"TP ({tp}) < LIMIT ({limit or stop or adjusted_price}) < SL ({sl})")

        order = Order(self, size, limit, stop, sl, tp, trade)
        if trade:
            self.orders.insert(0, order)
        else:
            self.orders.append(order)

        return order

    def _adjusted_price(self, size: int, price: Optional[float] = None) -> float:
        return (price or self.closing_price) * (1 + copysign(self.fee, size))

    def _process_orders(self):
        data = self.data
        open, high, low = data.Open[self.current_step], data.High[self.current_step], data.Low[self.current_step]
        prev_close = data.Close[self.current_step - 1]
        reprocess_orders = False

        # Process orders
        for order in list(self.orders):  # type: Order

            # Related SL/TP order was already removed
            if order not in self.orders:
                continue

            # Check if stop condition was hit
            stop_price = order.stop
            if stop_price:
                is_stop_hit = (high > stop_price) if order.is_long else (low < stop_price)
                if not is_stop_hit:
                    continue

                # > When the stop price is reached, a stop order becomes a market/limit order.
                # https://www.sec.gov/fast-answers/answersstopordhtm.html
                order._replace(stop_price=None)

            # Determine purchase price.
            # Check if limit order can be filled.
            if order.limit:
                is_limit_hit = low < order.limit if order.is_long else high > order.limit
                # When stop and limit are hit within the same bar, we pessimistically
                # assume limit was hit before the stop (i.e. "before it counts")
                is_limit_hit_before_stop = is_limit_hit and (
                    order.limit < (stop_price or -np.inf) if order.is_long else order.limit > (stop_price or np.inf)
                )
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue

                # stop_price, if set, was hit within this bar
                price = min(stop_price or open, order.limit) if order.is_long else max(stop_price or open, order.limit)
            else:
                # Market-if-touched / market order
                price = prev_close
                price = max(price, stop_price or -np.inf) if order.is_long else min(price, stop_price or np.inf)

            # Determine entry/exit bar index
            is_market_order = not order.limit and not stop_price
            time_index = (self.current_step - 1) if is_market_order else self.current_step

            # If order is a SL/TP order, it should close an existing trade it was contingent upon
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                # If order.size is "greater" than trade.size, this order is a trade.close()
                # order and part of the trade was already closed beforehand
                size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
                # If this trade isn't already closed (e.g. on multiple `trade.close(.5)` calls)
                if trade in self.trades:
                    self._reduce_trade(trade, price, size, time_index)
                    assert order.size != -_prev_size or trade not in self.trades
                if order in (trade._sl_order, trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders  # Removed when trade was closed
                else:
                    # It's a trade.close() order, now done
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                continue

            # Else this is a stand-alone trade

            # Adjust price to include commission (or bid-ask spread).
            # In long positions, the adjusted price is a fraction higher, and vice versa.
            adjusted_price = self._adjusted_price(order.size, price)

            # If order size was specified proportionally,
            # precompute true size in units, accounting for margin and spread/commissions
            size = order.size
            if -1 < size < 1:
                size = copysign(int((self.margin_available * self._leverage * abs(size)) // adjusted_price), size)
                # Not enough cash/margin even for a single unit
                if not size:
                    self.orders.remove(order)
                    continue
            assert size == round(size)
            need_size = int(size)

            if not self._hedging:
                # Fill position by FIFO closing/reducing existing opposite-facing trades.
                # Existing trades are closed at unadjusted price, because the adjustment
                # was already made when buying.
                for trade in list(self.trades):
                    if trade.is_long == order.is_long:
                        continue
                    assert trade.size * order.size < 0

                    # Order size greater than this opposite-directed existing trade,
                    # so it will be closed completely
                    if abs(need_size) >= abs(trade.size):
                        self._close_trade(trade, price, time_index)
                        need_size += trade.size
                    else:
                        # The existing trade is larger than the new order,
                        # so it will only be closed partially
                        self._reduce_trade(trade, price, need_size, time_index)
                        need_size = 0

                    if not need_size:
                        break

            # If we don't have enough liquidity to cover for the order, cancel it
            if abs(need_size) * adjusted_price > self.margin_available * self._leverage:
                self.orders.remove(order)
                continue

            # Open a new trade
            if need_size:
                self._open_trade(adjusted_price, need_size, order.sl, order.tp, time_index)

                # We need to reprocess the SL/TP orders newly added to the queue.
                # This allows e.g. SL hitting in the same bar the order was open.
                # See https://github.com/kernc/backtesting.py/issues/119
                if order.sl or order.tp:
                    if is_market_order:
                        reprocess_orders = True
                    elif low <= (order.sl or -np.inf) <= high or low <= (order.tp or -np.inf) <= high:
                        warnings.warn(
                            f"({data.index[-1]}) A contingent SL/TP order would execute in the "
                            "same bar its parent stop/limit order was turned into a trade. "
                            "Since we can't assert the precise intra-candle "
                            "price movement, the affected SL/TP order will instead be executed on "
                            "the next (matching) price/bar, making the result (of this trade) "
                            "somewhat dubious. "
                            "See https://github.com/kernc/backtesting.py/issues/119",
                            UserWarning,
                        )

            # Order processed
            self.orders.remove(order)

        if reprocess_orders:
            self._process_orders()

    def _reduce_trade(self, trade: Trade, price: float, size: float, time_index: int):
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0
        if not size_left:
            close_trade = trade
        else:
            # Reduce existing trade ...
            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)

            # ... by closing a reduced copy of it
            close_trade = trade._copy(size=-size, sl_order=None, tp_order=None)
            self.trades.append(close_trade)

        self._close_trade(close_trade, price, time_index)

    def _close_trade(self, trade: Trade, price: float, time_index: int):
        self.trades.remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        self.closed_trades.append(trade._replace(exit_price=price, exit_bar=time_index))
        self.assets += trade.pnl

    def _open_trade(self, price: float, size: int, sl: float, tp: float, time_index: int):
        trade = Trade(self, size, price, time_index)
        self.trades.append(trade)
        if tp:
            trade.tp = tp
        if sl:
            trade.sl = sl

    @property
    def next_observation(self) -> np.ndarray:
        next_single_obs = np.hstack((self.features.iloc[self.current_step - 1, :], self.position.pnl_pct))
        next_observation = np.vstack((self.observation[1:], next_single_obs))
        return next_observation

    @property
    def closing_price(self) -> float:
        return self.data["Close"][self.current_step]

    @property
    def current_time(self):
        return self.data.index[self.current_step]

    @property
    def tech_indicators(self):
        return self.features.columns.tolist()

    @property
    def equity(self) -> float:
        return self.assets + sum(trade.pnl for trade in self.trades)

    @property
    def margin_available(self) -> float:
        margin_used = sum(trade.value / self._leverage for trade in self.trades)
        return max(0, self.equity - margin_used)

    @property
    def sl_price(self) -> Optional[float]:
        if self.stop_loss == False:
            return None

        m = self.data["Low"][self.current_step - self.window_size : self.current_step + 1].min()
        M = self.data["High"][self.current_step - self.window_size : self.current_step + 1].max()
        if self.action == self.actions.Sell.value:
            sl = M * (1 + 0.01)
        elif self.action == self.actions.Buy.value:
            sl = m * (1 - 0.01)
        else:
            sl = None

        return sl
