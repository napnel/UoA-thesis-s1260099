from enum import Enum
import warnings
from bokeh.util.warnings import BokehDeprecationWarning

warnings.simplefilter("ignore", BokehDeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
from backtesting import Strategy, Backtest


class Actions(Enum):
    Sell = 0
    Buy = 1


class DRLStrategy(Strategy):
    model = None
    env = None

    def init(self):
        self.state = self.env.reset()
        self.max_step = len(self.data.df) - 1

    def next(self):
        self.step = len(self.data.df) - 1
        self.env.current_step = self.step  # BacktestのステップとEnvironmentのステップを同期させる

        if self.step < self.env.window_size:
            return

        assert self._broker._cash == self.env.wallet.assets, f"Step:{self.step}/{self.max_step}: {self._broker._cash} != {self.env.wallet.assets}"
        assert self.equity == self.env.wallet.equity, f"Step{self.step}/{self.max_step}: {self.equity} != {self.env.wallet.equity}"

        if self.step == self.max_step:
            self.env.position.close()
            return

        action, _ = self.model.predict(self.env.next_observation)

        if action == Actions.Buy.value and not self.position.is_long:
            if self.position.is_short:
                self.position.close()
            else:
                self.buy()
            self.env.buy()

        elif action == Actions.Sell.value and not self.position.is_short:
            if self.position.is_long:
                self.position.close()
            else:
                self.sell()
            self.env.sell()


def backtest(model, env, plot=False, plot_filename=None):
    bt = Backtest(env._df, DRLStrategy, cash=env.wallet.initial_assets, commission=env.fee, trade_on_close=True, exclusive_orders=False)
    stats = bt.run(model=model, env=env)
    if plot:
        bt.plot(filename=plot_filename)
    return stats
