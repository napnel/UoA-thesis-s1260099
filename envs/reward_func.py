from math import copysign


def profit_per_trade_reward(env):
    reward = 0
    if not env.closed_trades.empty and env.position.size == 0:
        if env.closed_trades.iloc[-1]["Steps"] == env.current_step - 1:
            reward = env.closed_trades.iloc[-1]["ReturnPct"] * 100

    return reward


def profit_per_tick_reward(env):
    reward = env.position.profit_or_loss_pct
    return reward


def shape_ratio_reward(env):
    # reward = 0.0
    if env.position.size == 0 or env.position.entry_time == env.current_datetime:
        print("Position size == 0")
        return 0.0

    trade_return = env._df.loc[env.position.entry_time : env.current_datetime, "Close"].pct_change().dropna().values
    print("Time: ", env.position.entry_time, " | ", env.current_datetime)
    print(env._df.loc[env.position.entry_time : env.current_datetime, "Close"])
    sharpe_ratio = copysign(1, env.position.size) * (trade_return.mean() / trade_return.std())
    print(f"Mean: {trade_return.mean()}, Std: {trade_return.std()}, SR: {sharpe_ratio}", "\n")
    return sharpe_ratio
