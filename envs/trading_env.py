import pandas as pd

from .base import BaseTradingEnv


class SimpleTradingEnv(BaseTradingEnv):
    def __init__(self, df: pd.DataFrame, features: pd.DataFrame, window_size: int, fee: float):
        super().__init__(df, features, window_size, fee)

    def _calculate_reward(self):
        reward = 0
        if self.closed_trades.empty:
            return reward

        trade = self.closed_trades.iloc[-1, :]
        return reward if trade["ExitTime"] != self.current_datetime else trade["ReturnPct"]
