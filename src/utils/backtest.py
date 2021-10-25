import os
import warnings
import pandas as pd
from typing import Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from backtesting import Strategy, Backtest


class DRLStrategy(Strategy):
    env = None
    agent = None

    def init(self):
        self.observation = self.env.reset()
        self.done = False

    def next(self):
        if self.data.index[-1] != self.env.current_time or self.done:
            pass

        else:
            assert self.data.Close[-1] == self.env.closing_price, self.debug()
            assert self._broker._cash == self.env.wallet.assets, self.debug()
            assert self.equity == self.env.wallet.equity, self.debug()
            # action, _ = self.agent.predict(self.env.observation, deterministic=True)
            if self.agent == "Random":
                action = self.env.action_space.sample()
            elif self.agent == "Buy&Hold":
                action = self.env.actions.Buy.value
            elif self.agent:
                action = self.agent.compute_single_action(self.env.observation, explore=False)
            # do Trade

            exec_trade = []

            if self.env.next_done:
                self.position.close()

            elif action == self.env.actions.Buy.value and not self.position.is_long:
                if self.position.is_short:
                    self.position.close()
                else:
                    exec_trade.append(self.buy)

            elif action == self.env.actions.Sell.value and not self.position.is_short:
                if self.position.is_long:
                    self.position.close()
                else:
                    exec_trade.append(self.sell)

            self.observation, _, self.done, _ = self.env.step(action)
            for trade in exec_trade:
                if self.env.position.size != 0:
                    trade(size=abs(self.env.position.size))

    def debug(self):
        print("===" * 10, "DEBUG", "===" * 10)
        print("Env Step: ", self.env.current_step)
        print("Env Price: ", self.env.closing_price, "| Backtest Price: ", self.data.Close[-1])
        print("Env Equity: ", self.env.wallet.equity, "| Backtest Equity: ", self.equity)
        print("Env Assets: ", self.env.wallet.assets, "| Backtest Assets: ", self._broker._cash)
        print("Env Position: ", self.env.position, "| Backtest Position: ", self.position)
        print("Backtest Trades: ", self.trades)
        print("===" * 10, "=====", "===" * 10)
        return "See Debug Above"


def backtest(
    env,
    agent="Random",
    save_dir: Optional[str] = None,
    plot: bool = True,
) -> pd.DataFrame:

    bt = Backtest(
        env.df,
        DRLStrategy,
        cash=env.wallet.initial_assets,
        commission=env.fee,
        trade_on_close=True,
    )
    stats = bt.run(env=env, agent=agent)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        performance = stats.loc[
            [
                "Start",
                "End",
                "Duration",
                "Return [%]",
                "Buy & Hold Return [%]",
                "Return (Ann.) [%]",
                "Volatility (Ann.) [%]",
                "Sharpe Ratio",
                "Max. Drawdown [%]",
            ]
        ]
        equity_curve = stats.loc["_equity_curve"]
        trades = stats.loc["_trades"]

        performance.to_csv(os.path.join(save_dir, "performance.csv"), header=False)
        equity_curve.to_csv(os.path.join(save_dir, "equity_curve.csv"))
        trades.to_csv(os.path.join(save_dir, "trades.csv"), index=False)
        if plot:
            bt.plot(filename=os.path.join(save_dir, "backtest"))

    return stats
