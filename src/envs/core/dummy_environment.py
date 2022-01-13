import sys
from enum import IntEnum
from typing import Callable, Optional, Tuple

import gym
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int,
        fee: float,
        reward_func: Callable,
        actions: IntEnum,
        stop_loss: bool,
        debug: bool,
    ):
        pass

    def reset(self) -> np.ndarray:
        pass

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        pass

    def render(self, mode="human"):
        pass

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
        pass

    def sell(
        self,
        size: Optional[float] = _FULL_EQUITY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        sl: float = None,
        tp: float = None,
    ):
        pass

    def new_order(
        self,
        size: float,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
        *,
        trade=None,
    ):
        pass

    def _adjusted_price(self, size: int, price: Optional[float] = None) -> float:
        pass

    def _process_orders(self):
        pass

    def _reduce_trade(self, trade, price: float, size: float, time_index: int):
        pass

    def _close_trade(self, trade, price: float, time_index: int):
        pass

    def _open_trade(
        self, price: float, size: int, sl: float, tp: float, time_index: int
    ):
        pass

    @property
    def next_observation(self) -> np.ndarray:
        pass

    @property
    def closing_price(self) -> float:
        pass

    @property
    def current_time(self):
        pass

    @property
    def tech_indicators(self):
        pass

    @property
    def equity(self) -> float:
        pass

    @property
    def margin_available(self) -> float:
        pass

    @property
    def sl_price(self) -> Optional[float]:
        pass
