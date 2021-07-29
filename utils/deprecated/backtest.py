import warnings
from bokeh.util.warnings import BokehDeprecationWarning

warnings.simplefilter("ignore", BokehDeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
from backtesting import Strategy, Backtest


class DRLStrategy(Strategy):
    model = None
    env = None

    def init(self):
        self.state = self.env.reset()
        self.max_step = len(self.data.df) - 1

    def next(self):
        self.step = len(self.data.df) - 1
        self.env.current_step = self.step  # BacktestのステップとEnvironmentのステップを同期させる
        self.env.broker.current_step = self.step

        if self.step < self.env.lookback_window or self.env.is_terminal:
            return

        self.env.update_state()
        assert self._broker._cash == self.env.broker.assets, f"Step:{self.step}/{self.max_step}: {self._broker._cash} != {self.env.broker.assets}"
        assert self.equity == self.env.broker.equity, f"Step{self.step}/{self.max_step}: {self.equity} != {self.env.broker.equity}"

        if self.step + 1 == self.max_step:
            self.env.broker.position.close()

        action, _ = self.model.predict(self.env.state)
        if action == 0:
            pass

        elif action == 1 and not self.position.is_long:
            if self.position.is_short:
                self.position.close()
            else:
                size = self._broker.margin_available // (self._broker.last_price * (1 + self._broker._commission))
                if size != 0:
                    self.buy(size=size)
            self.env.buy()

        elif action == 2 and not self.position.is_short:
            if self.position.is_long:
                self.position.close()
            else:
                size = self._broker.margin_available // (self._broker.last_price * (1 + self._broker._commission))
                if size != 0:
                    self.sell(size=size)
            self.env.sell()


def backtest(model, env, assets, fee, plot=False, plot_filename=None):
    bt = Backtest(env._df, DRLStrategy, cash=assets, commission=fee, trade_on_close=True, exclusive_orders=False)
    stats = bt.run(model=model, env=env)
    if plot:
        bt.plot(filename=plot_filename)
    return stats
