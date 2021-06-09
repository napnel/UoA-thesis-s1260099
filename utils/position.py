class Position:
    def __init__(self, size=None, entry_price=None):
        self.__size = size
        self.__entry_price = entry_price

    def __repr__(self) -> str:
        return f"Position(size: {self.size}, entry_price: {self.entry_price})"

    def is_long(self) -> bool:
        return True if self.__size > 0 else False

    def is_short(self) -> bool:
        return True if self.__size < 0 else False

    @property
    def size(self):
        return self.__size

    @property
    def entry_price(self):
        return self.__entry_price

    @size.setter
    def size(self, size):
        self.__size = size

    @entry_price.setter
    def entry_price(self, price):
        self.__entry_price = price


def test():
    pass


if __name__ == "__main__":
    test()
