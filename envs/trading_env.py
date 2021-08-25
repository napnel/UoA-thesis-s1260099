from .base_env import BaseTradingEnv


class TradingEnv(BaseTradingEnv):
    def _calculate_reward(self):
        # reward = 0.0
        reward = self.position.profit_or_loss_pct
        if self.closed_trades.empty:
            return reward

        trade = self.closed_trades.iloc[-1, :]
        return reward if trade["ExitTime"] != self.current_datetime else trade["ReturnPct"]
