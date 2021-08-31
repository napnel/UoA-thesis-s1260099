from math import copysign
from .base_env import BaseTradingEnv


class TradingEnv(BaseTradingEnv):
    """Profit Per Tick"""

    def _calculate_reward(self):
        reward = self.position.profit_or_loss_pct
        if self.closed_trades.empty:
            return reward

        if self.closed_trades.iloc[-1]["ExitTime"] == self.current_datetime:
            reward = self.closed_trades.iloc[-1]["ReturnPct"]

        return reward


class SRTradingEnv(BaseTradingEnv):
    def _calculate_reward(self):
        # reward = 0.0
        if self.position.size == 0 or self.position.entry_time == self.current_datetime:
            print("Position size == 0")
            return 0.0

        trade_return = self._df.loc[self.position.entry_time : self.current_datetime, "Close"].pct_change().dropna().values
        print("Time: ", self.position.entry_time, " | ", self.current_datetime)
        print(self._df.loc[self.position.entry_time : self.current_datetime, "Close"])
        sharpe_ratio = copysign(1, self.position.size) * (trade_return.mean() / trade_return.std())
        print(f"Mean: {trade_return.mean()}, Std: {trade_return.std()}, SR: {sharpe_ratio}", "\n")
        return sharpe_ratio
