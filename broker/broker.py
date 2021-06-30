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

    def new_order(self, size, limit_price):
        if size != 0:
            if self.position.size == 0:
                self.position.size = size
                self.position.entry_price = limit_price
            else:
                assert self.position.size == -size, f"position don't close: {self.position.size}, {size}"
                self.position.close()

    def get_candles(self, start, end) -> np.ndarray:
        return self.data.iloc[start:end, :].values

    @property
    def free_assets(self):
        return self.assets - abs(self.position.size) * self.current_price

    @property
    def equity(self):
        return self.assets + self.position.profit_or_loss

    @property
    def latest_candle(self) -> np.ndarray:
        return self.data.iloc[self.current_step, :].values

    @property
    def current_datetime(self):
        return self.data.iloc[self.current_step, :].name

    @property
    def current_price(self) -> np.ndarray:
        return self.data["Close"][self.current_step]

    @property
    def account_state(self) -> np.ndarray:
        return np.array([self.assets, self.position.profit_or_loss])


class Position:
    def __init__(self, broker: Broker):
        self.__broker = broker
        self.__size = 0
        self.__entry_price = None

    def __repr__(self) -> str:
        return f"Position(size: {self.size}, entry_price: {self.entry_price}, pl: {self.profit_or_loss:.0f})"

    @property
    def size(self) -> float:
        return self.__size

    @property
    def entry_price(self) -> float:
        return self.__entry_price

    @size.setter
    def size(self, size):
        self.__size = size

    @entry_price.setter
    def entry_price(self, price):
        self.__entry_price = price

    @property
    def is_long(self) -> bool:
        return True if self.__size > 0 else False

    @property
    def is_short(self) -> bool:
        return True if self.__size < 0 else False

    @property
    def profit_or_loss(self):
        if self.__size == 0:
            return 0
        return self.__size * (self.__broker.current_price - self.__entry_price)

    def close(self):
        self.__broker.assets += self.profit_or_loss
        self.__size = 0
        self.__entry_price = None
