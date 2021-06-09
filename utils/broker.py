import pandas as pd

from typing import List


class Broker:
    def __init__(self, data: pd.DataFrame, net_worth: float, fee: float):
        self.data = data
        self.current_index = 0
        self.net_worth = net_worth
        self.fee = fee
        self.position: Position = Position(self)

    def new_order(self):
        pass

    def next(self):
        self.current_index += 1

    @property
    def latest_candle(self):
        return self.data[self.current_index]

    @property
    def latest_price(self):
        return self.data["Close"][self.current_index]


class Position:
    def __init__(self, broker: Broker):
        self.__broker = broker
        self.__size = None
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
        return self.__size * (self.__broker.last_price - self.__entry_price)

    def close(self):
        self.__size = None
        self.__entry_price = None
        self.__broker.net_worth += self.profit_or_loss
