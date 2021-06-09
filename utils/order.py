class Order:
    def __init__(self, symbol: str, size: float, price: float):
        assert size != 0
        self.__symbol = symbol
        self.__size = size
        self.__price = price

    def __repr__(self) -> str:
        return f"Order(symbol: {self.symbol}, size: {self.size}, price: {self.price})"

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def side(self) -> str:
        return "buy" if self.__size > 0 else "sell"

    @property
    def size(self) -> float:
        return self.__size

    @property
    def price(self) -> float:
        return self.__price


def main():
    order = Order("BTCUSDT", 1, 40)
    print(order)


if __name__ == "__main__":
    main()
