import argparse
import glob
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from empyrical import annual_return, max_drawdown, sharpe_ratio

pd.options.display.float_format = "{:.2f}".format

parser = argparse.ArgumentParser()
parser.add_argument("--local_dir", type=str, default="./ray_results")
args = parser.parse_args()
print(args)


def summary_computation_time(local_dir: str):
    algo_expt_paths = sorted(glob.glob(os.path.join(local_dir, "*__*")))
    n_fold, n_cpus, n_workers = 5, 8, 2
    n_parallel = n_cpus / n_workers

    total_time = 0
    total_time_df = pd.DataFrame()
    print(algo_expt_paths)
    for expt_path in algo_expt_paths:
        algo = expt_path.split("__")[0][-3:]
        print("===" * 15, algo, "===" * 15)
        sample_paths = glob.glob(os.path.join(expt_path, "ExperimentCV-*"))
        total_time_per_algo = 0
        n_samples = len(sample_paths) / n_fold
        print(sample_paths)
        for sample_path in sample_paths:
            progress = pd.read_csv(os.path.join(sample_path, "progress.csv"))
            progress = progress[["timesteps_total", "time_this_iter_s"]]
            progress = progress[progress["timesteps_total"] <= 150000]
            total_time_per_algo += progress["time_this_iter_s"].sum()

        times = pd.Series(
            [
                total_time_per_algo / 60 / n_samples,
                total_time_per_algo / 60 / n_samples / n_fold,
            ],
            index=["Per CV [min]", "Per Trial [min]"],
            name=algo,
        )
        total_time_df = pd.concat([total_time_df, times], axis=1)
        total_time += total_time_per_algo * 4

    sum_total_time = total_time_df.sum(axis=1)
    sum_total_time.name = "Sum"
    total_time_df = pd.concat([total_time_df, sum_total_time], axis=1)
    total_time_df.to_csv(os.path.join(local_dir, "times.csv"), float_format="%.1f")
    print(total_time_df)
    print(f"Our machine takes {total_time / 60 / 60 / n_parallel:.2f} [hours]")


def summary_learning_curve(local_dir: str):
    algo_expt_paths = sorted(glob.glob(os.path.join(local_dir, "*__*")))

    def get_bh_reward(data_type: str):
        bh_dir = os.path.join(local_dir, "backtest-stats-buy&hold/test-*")
        bh_folder_cv = sorted(glob.glob(bh_dir))
        bh_reward_cv = []
        for bh_folder in bh_folder_cv:
            equity_curve = pd.read_csv(
                os.path.join(bh_folder, "equity_curve.csv"), index_col=0
            )["Equity"]
            reward = equity_curve.apply(np.log).diff().dropna().sum()
            bh_reward_cv.append(reward)

        bh_reward_mean = sum(bh_reward_cv) / len(bh_reward_cv)
        return bh_reward_mean

    with plt.style.context(["science", "ieee"]):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        min_last_timestep = 1e9
        max_first_timestep = 0
        timesteps = None
        for expt_path in algo_expt_paths:
            algo = expt_path.split("__")[0][-3:]

            with open(os.path.join(expt_path, "results.pkl"), "rb") as f:
                results: pd.DataFrame = pickle.load(f)

            params = list(results.index.names)
            params.remove("timesteps_total")
            alpha = 0.3
            # Get best results with considering reward condition
            bh_reward_train = get_bh_reward("train")
            # results_filtered = results.query("episode_reward_mean > @bh_reward_train")
            results_filtered = results.query("episode_reward_mean > -1")
            top_train = (
                results_filtered["episode_reward_mean"]
                .ewm(alpha=alpha)
                .mean()
                .groupby(params)
                .last()
                .sort_values(ascending=False)
            )
            top_valid = (
                results_filtered["evaluation/episode_reward_mean"]
                .ewm(alpha=alpha)
                .mean()
                .groupby(params)
                .last()
                .sort_values(ascending=False)
            )
            best_config_train = dict(zip(params, top_train.index[0]))
            best_results_train = results.loc[tuple(best_config_train.values())]
            best_config_valid = dict(zip(params, top_valid.index[0]))
            best_results_valid = results.loc[tuple(best_config_valid.values())]

            # Plot
            timesteps = best_results_train.index.values
            min_last_timestep = min(min_last_timestep, timesteps[-1])
            max_first_timestep = max(max_first_timestep, timesteps[0])
            training_reward = (
                best_results_train["episode_reward_mean"].ewm(alpha=alpha).mean()
            )
            validation_reward = (
                best_results_valid["evaluation/episode_reward_mean"]
                .ewm(alpha=alpha)
                .mean()
            )
            axes[0].plot(timesteps, training_reward, label=algo)
            axes[1].plot(timesteps, validation_reward, label=algo)

        bh_reward_train = pd.Series(
            [get_bh_reward("train")] * len(timesteps),
            index=training_reward.index,
        )
        bh_reward_eval = pd.Series(
            [get_bh_reward("eval")] * len(timesteps),
            index=validation_reward.index,
        )
        axes[0].plot(
            timesteps, bh_reward_train, label="B\&H", linestyle=(10, (5, 3, 1, 3, 1, 3))
        )
        axes[1].plot(
            timesteps, bh_reward_eval, label="B\&H", linestyle=(10, (5, 3, 1, 3, 1, 3))
        )

        axes[0].set_title("Training")
        axes[1].set_title("Validation")
        axes[0].set_ylabel("Reward")
        axes[0].set_xlabel("Timesteps")
        axes[1].set_xlabel("Timesteps")
        axes[0].set_xlim(max_first_timestep, min_last_timestep)
        axes[1].set_xlim(max_first_timestep, min_last_timestep)
        axes[0].legend(loc="upper left")
        axes[1].legend(loc="upper left")

        local_dir = str(Path(algo_expt_paths[0]).parent)
        plt.savefig(os.path.join(local_dir, "Learning Curve"))
        plt.close("all")


def get_performance_from_equity(local_dir: str):
    algo_expt_paths = sorted(glob.glob(os.path.join(local_dir, "*__*")))
    algo_avg_performance = pd.DataFrame()

    def calc_performance(equity_curve: pd.Series, name: str = None):
        returns = equity_curve.pct_change()
        ann_return = annual_return(returns) * 100
        ann_sharpe_ratio = sharpe_ratio(returns, annualization=True)
        ann_max_drawdown = max_drawdown(returns) * 100
        index = ["Cum Return [%]", "Max. Drawdown [%]", "Sharpe Ratio"]
        performance = pd.Series(
            [ann_return, ann_max_drawdown, ann_sharpe_ratio], index=index, name=name
        )
        return performance

    for expt_path in algo_expt_paths:
        algo = expt_path.split("__")[0][-3:]
        print("===" * 15, algo, "===" * 15)
        backtest_paths = glob.glob(os.path.join(expt_path, "backtest-stats-test*"))
        backtest_paths = sorted(backtest_paths)
        all_performance = pd.DataFrame()
        for backtest_path in backtest_paths:
            equity_curve = pd.read_csv(
                os.path.join(backtest_path, "equity_curve.csv"), index_col=0
            )["Equity"]
            performance = calc_performance(equity_curve, backtest_path[-1])
            all_performance = pd.concat([all_performance, performance], axis=1)

        avg_performance = pd.Series(
            np.nanmean(all_performance, axis=1), index=all_performance.index
        )
        avg_performance.name = "Avg"
        all_performance = pd.concat([all_performance, avg_performance], axis=1)
        print(all_performance)
        avg_performance.name = algo
        algo_avg_performance = pd.concat(
            [algo_avg_performance, avg_performance], axis=1
        )

    bh_dir = os.path.join(local_dir, "backtest-stats-buy&hold/test-*")
    bh_folder_cv = sorted(glob.glob(bh_dir))
    all_bh_performance = pd.DataFrame()
    for bh_folder in bh_folder_cv:
        equity_curve = pd.read_csv(
            os.path.join(bh_folder, "equity_curve.csv"), index_col=0
        )["Equity"]
        performance = calc_performance(equity_curve, name=bh_folder[-1])
        all_bh_performance = pd.concat([all_bh_performance, performance], axis=1)

    avg_bh_performance = all_bh_performance.mean(axis=1)
    avg_bh_performance.name = "Avg"
    all_bh_performance = pd.concat([all_bh_performance, avg_bh_performance], axis=1)
    print("===" * 15, "B&H", "===" * 15)
    print(all_bh_performance)

    avg_bh_performance.name = "B&H"
    algo_avg_performance = pd.concat([algo_avg_performance, avg_bh_performance], axis=1)
    print("===" * 15, f"Overall Performance", "===" * 15)
    print(algo_avg_performance.T)
    algo_avg_performance.T.to_csv(
        os.path.join(local_dir, "avg_performance.csv"), float_format="%.2f"
    )


if __name__ == "__main__":
    local_dir = Path(args.local_dir).resolve()
    local_dir = os.path.join(local_dir)
    summary_computation_time(local_dir)
    summary_learning_curve(local_dir)
    get_performance_from_equity(local_dir)
