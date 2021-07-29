import pandas as pd

from .base import BaseTradingEnv


class SimpleTradingEnv(BaseTradingEnv):
    def __init__(self, df: pd.DataFrame, preprocessed_df: pd.DataFrame, window_size: int, fee: float):
        super().__init__(df, preprocessed_df, window_size, fee)

    def _calculate_reward(self):
        reward = self.position.profit_or_loss_pct - self.prev_profit_or_loss_pct * 100
        self.prev_profit_or_loss_pct = self.position.profit_or_loss_pct
        return reward
