import os
import time
import pandas as pd
from datetime import datetime

from ftx import FtxClient
from dotenv import load_dotenv

# from utils import Position, Order, Wallet


class FTXAPI:
    def __init__(self):
        super().__init__()
        load_dotenv(verbose=True)
        dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(dotenv_path)
        self.api_key = os.environ["FTX_API_KEY"]
        self.api_secret = os.environ["FTX_API_SECRET"]
        self.client = FtxClient(api_key=self.api_key, api_secret=self.api_secret)

    def fetch_candle(self, symbol: str, interval: int, limit: int) -> pd.DataFrame:
        df = pd.DataFrame()
        start_time = time.time() - interval * limit
        end_time = time.time()
        candles = self.client.get_historical_data(symbol, interval, limit, start_time, end_time)
        for candle in candles:
            candle = pd.DataFrame.from_dict(candle, orient="index").T
            candle = candle.drop(["startTime"], axis=1)
            df = pd.concat([df, candle], axis=0)

        df["time"] = df["time"] / 1000  # timestamp ns -> ms
        df = df.rename(columns={"time": "Date"}).set_index("Date")
        df = df.rename(columns=str.capitalize, index=datetime.fromtimestamp)
        df = df.astype(float)
        return df

    # def get_position(self, symbol: str) -> Position:
    #     response = self.client.get_position(symbol)
    #     size = response["size"] if response["side"] == "buy" else -response["size"]
    #     position = Position(size=size, entry_price=response["recentAverageOpenPrice"])
    #     return position

    # def get_wallet(self) -> Wallet:
    #     response = self.client.get_account_info()
    #     wallet = Wallet(collateral=response["collateral"], free_collateral=response["freeCollateral"], leverage=response["leverage"])
    #     return wallet

    # def place_order(self, order: Order):
    #     response = self.client.place_order(order.symbol, order.side.lower(), order.price, order.size, "limit")
    #     return response

    # def conditional_order(self, order: Order, type: str, trigger_price):
    #     response = self.client.place_conditional_order(order.symbol, order.side.lower(), order.size, type, order.price, trigger_price=trigger_price)
    #     return response

    # def cancel_all_order(self, symbol: str):
    #     response = self.client.cancel_orders(symbol)
    #     return response

    # def get_trade_records(self):
    #     return super().get_trade_records()

    # def get_orderbook(self, symbol, depth=10):
    #     response = self.client.get_orderbook(symbol, depth)
    #     return response
