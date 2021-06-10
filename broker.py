import numpy as np
import pandas as pd

from typing import List


class Broker:
    def __init__(self, data: pd.DataFrame, assets: float, fee: float):
        self.data: pd.DataFrame = data
        self.current_index = 0
        self.assets = assets
        self.fee = fee
        self.position: Position = Position(self)

    def buy(self, size, limit_price):
        return self.new_order(size, limit_price)

    def sell(self, size, limit_price):
        return self.new_order(-size, limit_price)

    def new_order(self, size, limit_price):
        self.position.size = size
        self.position.entry_price = limit_price

    def get_candles(self, start, end) -> np.ndarray:
        return self.data.iloc[start:end, :].values

    @property
    def latest_candle(self) -> np.ndarray:
        return self.data[self.current_index].values

    @property
    def current_price(self) -> np.ndarray:
        return self.data["Close"][self.current_index]


class Position:
    def __init__(self, broker: Broker):
        self.__broker = broker
        self.__size = 0
        self.__entry_price = None

    def __repr__(self) -> str:
        return f"Position(size: {self.size}, entry_price: {self.entry_price})"

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
        self.__size = 0
        self.__entry_price = None
        self.__broker.assets += self.profit_or_loss
