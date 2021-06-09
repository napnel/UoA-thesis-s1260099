# 学習後にバックテストをする


import gym
from backtesting import Strategy, Backtest
from stable_baselines3 import A2C

from ftx_api import FTXAPI

class TradingEnv(gym.Env):

    def __init__(self, df: pd.DataFrame, window_size: int, assets: float):
        self.df = df
        self.window_size = window_size
        self.action_space = spaces.Discrete(3)
        self.state_size = 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.window_size * self.state_size,))
        
        self.current_step: int = 0

        self.initial_assets = assets
        self.coin_bought: float = None

    def reset(self):
        self.wallet = Wallet(collateral=self.initial_assets, free_collateral=self.initial_assets, leverage=1)
        self.position = Position()
        self.current_step = 0
        self.coin_bought = 0.0

        state = self.df.iloc[:self.window_size].values
        state = state.reshape(self.window_size,)
        return state

    def step(self, action):
        '''
        action: 0 -> Hold, 1 -> Buy, 2 -> Sell
        '''
        self.price = self.df.iloc[self.window_size + self.current_step, :].values
        self.price = np.squeeze(self.price)

        obs = self.df.iloc[self.current_step:self.window_size + self.current_step].values
        obs = obs.reshape(self.window_size,)

        if action == 0:
            pass

        elif action == "buy" and not self.position.is_long:
            self.position.size += (self.wallet.free_collateral * self.wallet.leverage) / self.price
            self.position.entry_price = self.price
            self.wallet.free_collateral -= self.position.size * self.position.entry_price

        elif action == "sell" and not self.position.is_short:
            self.position.size -= ()
            self.balance -= self.price * self.coin_bought
            self.coin_bought = 0.0

        reward = self.balance if action > 0 else 0

        done = False
        if self.window_size + self.current_step + 1 == len(self.df):
            done = True

        info = {}
        self.current_step += 1
        
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Total Step: {self.current_step}")
        print(f"Balance: {self.balance}, Amount Of Coin: {self.coin_bought}\n")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


class RLStrategy(Strategy):
    
    def init(self):
        self.env = TradingEnv()
        self.model = A2C('MlpPolicy', self.env, verbose=1)

        self.obs = self.env.reset()
        self.state = None


    def next(self):
        action, state = self.model.predict(self.obs, state=self.state, deterministic=True)
        self.obs, reward, done, info = self.env.step(action)


def main():
    ftx = FTXAPI()
    df = ftx.fetch_candle("ETH-PERP", interval=15 * 60, limit=672)
    bt = Backtest(df, RLStrategy)
    stats = bt.run()
    print(stats)


if __name__ == '__main__':
    main()