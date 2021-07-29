from math import copysign
import numpy as np
import pandas as pd

from typing import List, Optional


class Broker:
    def __init__(self, data: pd.DataFrame, assets: float, fee: float):
        self.data: pd.DataFrame = data
        self.current_step = 0
        self.assets = assets
        self.fee = fee
        self.position: Position = Position(self)
        self.closed_trades = pd.DataFrame(columns=["size", "entry_price", "exit_price", "PnL", "entry_time", "exit_time"])

    def new_order(self, size, limit_price):
        if size != 0:
            if self.position.size == 0:
                self.position.size = size
                self.position.entry_price = limit_price * (1 + copysign(self.fee, size))
                self.position.entry_time = self.data.index[self.current_step]
            else:
                assert self.position.size == -size, f"position don't close: {self.position.size}, {size}"
                self.position.close()

    @property
    def equity(self):
        return self.assets + self.position.profit_or_loss

    @property
    def free_assets(self):
        used_assets = abs(self.position.size) * self.current_price
        return max(0, self.equity - used_assets)

    @property
    def current_datetime(self):
        return self.data.iloc[self.current_step, :].name

    @property
    def current_price(self) -> np.ndarray:
        return self.data["Close"][self.current_step]

    @property
    def account_state(self) -> np.ndarray:
        return np.array([self.free_assets > (self.current_price * (1 + self.fee)), self.position.profit_or_loss_pct])


class Position:
    def __init__(self, broker: Broker):
        self.__broker = broker
        self.size = 0
        self.entry_price = None
        self.entry_time = None

    def __repr__(self) -> str:
        return f"Position(size: {self.size}, entry_price: {self.entry_price}, pl: {self.profit_or_loss:.0f})"

    @property
    def is_long(self) -> bool:
        return True if self.size > 0 else False

    @property
    def is_short(self) -> bool:
        return True if self.size < 0 else False

    @property
    def profit_or_loss(self):
        if self.size == 0:
            return 0
        return self.size * (self.__broker.current_price - self.entry_price)

    @property
    def profit_or_loss_pct(self):
        if self.size == 0:
            return 0
        return copysign(1, self.size) * (self.__broker.current_price - self.entry_price) / self.entry_price

    def close(self):
        self.__broker.assets += self.profit_or_loss
        trade = {
            "size": self.size,
            "entry_price": self.entry_price,
            "exit_price": self.__broker.current_price,
            "PnL": self.profit_or_loss,
            "entry_time": self.entry_time,
            "exit_time": self.__broker.current_datetime,
        }
        self.__broker.closed_trades = self.__broker.closed_trades.append(trade, ignore_index=True)
        self.size = 0
        self.entry_price = None
        self.entry_time = None
