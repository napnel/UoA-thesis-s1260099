import os
import warnings
import pandas as pd
from pprint import pprint
from typing import Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import backtesting
    from backtesting import Strategy, Backtest


class DRLStrategy(Strategy):
    env = None
    agent = None
    debug = False

    def init(self):
        self.observation = self.env.reset()
        self.done = False
        self.stop_loss = self.env.stop_loss

    def next(self):
        if self.data.index[-1] != self.env.current_time or self.done:
            pass

        else:
            assert self.data.Close[-1] == self.env.closing_price, self.error()
            assert self._broker._cash == self.env.assets, self.error()
            assert self.equity == self.env.equity, self.error()
            # action, _ = self.agent.predict(self.env.observation, deterministic=True)
            if self.agent == "Random":
                action = self.env.action_space.sample()
            elif self.agent == "Buy&Hold":
                action = 2 if len(self.env.actions) == 3 else 1
            elif self.agent == "Sell&Hold":
                action = 0 if len(self.env.actions) == 2 else 0
            elif self.agent:
                action = self.agent.compute_single_action(self.env.observation, explore=False)
            # do Trade
            self.env.action = action
            if self.env.done:
                self.position.close()

            else:
                self.env.actions.perform(self, action)

            if self.debug:
                self.render()

            self.observation, _, self.done, _ = self.env.step(action)

    def error(self):
        print("===" * 10, "DEBUG", "===" * 10)
        print("Env Step: ", self.env.current_step)
        print("Env Position: ", self.env.position, "| Backtest Position: ", self.position)
        print("Env Price: ", self.env.closing_price, "| Backtest Price: ", self.data.Close[-1])
        print("Env Equity: ", self.env.equity, "| Backtest Equity: ", self.equity)
        print("Env Assets: ", self.env.assets, "| Backtest Assets: ", self._broker._cash)
        print("===" * 10, "=====", "===" * 10)
        self.render()
        return "See Debug Above"

    def render(self):
        print("===" * 5, f"Backtesting ({self.data.index[-1]})", "===" * 5)
        print(f"Price: {self.data.Close[-1]}")
        print(f"Assets: {self._broker._cash}")
        print(f"Equity: {self.equity}")
        print(f"Orders: {self.orders}")
        print(f"Trades: {self.trades}")
        print(f"Position: {self.position}")
        print(f"Closed Trades: {self.closed_trades}")

    @property
    def trade_size(self):
        return self.env.trade_size

    @property
    def latest_high_price(self):
        return self.env.latest_high_price

    @property
    def latest_low_price(self):
        return self.env.latest_low_price


def backtest(
    env,
    agent="Random",
    save_dir: Optional[str] = None,
    plot: bool = True,
    open_browser: bool = True,
    debug: bool = False,
) -> pd.DataFrame:

    bt = Backtest(
        env.data,
        DRLStrategy,
        cash=env.initial_assets,
        commission=env.fee,
        trade_on_close=True,
        # hedging=True,
    )
    stats = bt.run(env=env, agent=agent, debug=debug)

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

        performance.to_csv(os.path.join(save_dir, "performance.csv"))
        equity_curve.to_csv(os.path.join(save_dir, "equity_curve.csv"))
        trades.to_csv(os.path.join(save_dir, "trades.csv"), index=False)
        if plot:
            bt.plot(
                filename=os.path.join(save_dir, "backtest"),
                superimpose=False,
                open_browser=open_browser,
                plot_return=True,
                plot_equity=False,
            )

    return stats
