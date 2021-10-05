import warnings

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from backtesting import Strategy, Backtest

RandomAgent = None


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
            if self.agent:
                action = self.agent.compute_single_action(self.env.observation, explore=False)
            else:
                action = self.env.action_space.sample()
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


def backtest(env, agent=RandomAgent, plot=False, plot_filename=None) -> pd.DataFrame:
    bt = Backtest(
        env.df,
        DRLStrategy,
        cash=env.wallet.initial_assets,
        commission=env.fee,
        trade_on_close=True,
    )
    stats = bt.run(env=env, agent=agent)
    if plot:
        bt.plot(filename=plot_filename)
    return stats
