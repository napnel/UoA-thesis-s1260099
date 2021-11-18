from math import copysign
from empyrical import sharpe_ratio
import numpy as np
import pandas as pd

from src.envs.core.dummy_environment import TradingEnv


def profit_per_trade_reward(env: TradingEnv):
    reward = 0
    if not env.closed_trades.empty and env.position.size == 0:
        if env.closed_trades.iloc[-1]["Steps"] == env.current_step - 1:
            reward = env.closed_trades.iloc[-1]["ReturnPct"] * 100

    return reward


def profit_per_tick_reward(env: TradingEnv):
    reward = env.position.pnl_pct
    return reward


def equity_log_return_reward(env: TradingEnv):
    reward = np.log(env.equity_curve[-1]) - np.log(env.equity_curve[-2])
    return reward


def initial_equity_return_reward(env: TradingEnv):
    reward = (env.equity_curve[-1] - env.equity_curve[0]) / env.equity_curve[0]
    return reward


def sharpe_ratio_reward(env: TradingEnv):
    # prev_sharpe_ratio_reward = env.reward
    reward = 0
    prev_returns = pd.Series(env.equity_curve[:-1]).pct_change().dropna()
    returns = pd.Series(env.equity_curve).pct_change().dropna()
    prev_sharpe = 0
    sharpe = 0
    if len(prev_returns) >= 2:
        prev_sharpe = sharpe_ratio(prev_returns)
        sharpe = sharpe_ratio(returns)
    # if not env.closed_trades.empty and env.position.size == 0:
    #     if env.closed_trades.iloc[-1]["Steps"] == env.current_step - 1:
    #         reward = env.closed_trades.iloc[-1]["SharpeRatio"]

    # if np.isnan(reward):
    #     reward = 0
    # reward = env.position.pnl_pct

    # if not env.closed_trades.empty and env.position.size == 0:
    #     trade = env.closed_trades.iloc[-1]
    #     if trade["Steps"] == env.current_step - 1:
    #         returns = copysign(1, trade["Size"]) * env.df.loc[trade["EntryTime"] : trade["ExitTime"], "Close"].pct_change()
    #         reward = sharpe_ratio(returns)

    # if prev_sharpe_ratio_reward != 0:
    #     reward = (reward - prev_sharpe_ratio_reward) / prev_sharpe_ratio_reward
    reward = sharpe - prev_sharpe
    return reward
