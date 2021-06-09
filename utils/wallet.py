class Wallet:
    def __init__(self, collateral, free_collateral, leverage):
        self.__collateral = collateral
        self.__free_collateral = free_collateral
        self.__leverage = leverage

    def __repr__(self) -> str:
        return f"Wallet(collateral: {self.collateral}, free_collateral: {self.free_collateral}, leverage: {self.leverage})"

    @property
    def collateral(self) -> float:
        return self.__collateral

    @property
    def free_collateral(self) -> float:
        return self.__free_collateral

    @property
    def leverage(self) -> float:
        return self.__leverage


def test():
    wallet = Wallet(100, 10, 1)
    print(wallet)


if __name__ == "__main__":
    test()
