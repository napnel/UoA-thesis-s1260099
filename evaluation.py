import os
import glob
import pathlib
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt

import ray
from ray import tune
from ray.tune import ExperimentAnalysis
from src.envs import TradingEnv
from src.utils import backtest
from src.trainable.cross_validation import ExperimentCV
from src.trainable.util import prepare_config_for_agent
from src.utils.tuning_space import get_tuning_params


def get_best_param_cv(analysis):
    pass


def get_epxt_perfomance(analysis: ExperimentAnalysis):
    """
    評価データにおける累積報酬が最も高かったパラメータのCVの投資性能を取得する。
    """
    trials = get_best_param_cv(analysis)
    trials_performance_cv = {}
    for trial in trials:
        with open(os.path.join(trial, "params.json"), "r") as f:
            params = json.load(f)
        train_start = datetime.strptime(params["_train_start"], "%Y-%m-%d") + relativedelta(years=params["__trial_index__"])
        eval_start = datetime.strptime(params["_train_start"], "%Y-%m-%d") + relativedelta(years=params["__trial_index__"] + params["_train_years"])
        column_name = f"{train_start.year}(+{params['_train_years']}) | {eval_start.year}(+{params['_eval_years']})"

        perfomance_train = pd.read_csv(os.path.join(trial, "last-stats-train", "performance.csv"), index_col=0).rename(columns={"0": column_name})
        perfomance_eval = pd.read_csv(os.path.join(trial, "last-stats-eval", "performance.csv"), index_col=0).rename(columns={"0": column_name})
        perfomance_test = pd.read_csv(os.path.join(trial, "last-stats-test", "performance.csv"), index_col=0).rename(columns={"0": column_name})
        trials_performance_cv[column_name] = perfomance_test

    performance_cv = pd.DataFrame()
    for trial, performance in trials_performance_cv.items():
        performance = performance.iloc[-4:, :].astype(float)
        performance_cv = pd.concat([performance_cv, performance], axis=1)

    performance_cv_mean = performance_cv.mean(axis=1)
    performance_cv_mean.to_csv(os.path.join(analysis._experiment_dir, "performance.csv"))
    return performance_cv_mean


def get_expt_results_cv(analysis: ExperimentAnalysis):
    all_configs = analysis.get_all_configs()
    expt_df = pd.DataFrame()
    parameter_table = pd.DataFrame()

    for logdir, df in analysis.trial_dataframes.items():
        # df = df[["episode_reward_mean", "evaluation/episode_reward_mean", "timesteps_total"]]
        config = all_configs[logdir].copy()
        for k, v in config["env_config"].items():
            config[f"env_config/{k}"] = v
        config.pop("env_config")
        tuned_params = list(get_tuning_params(config["_algo"]).keys())
        tuned_config = {k: v for k, v in config.items() if k in tuned_params}
        tuned_config = pd.DataFrame(tuned_config, index=[0])
        parameter_table = pd.concat([parameter_table, tuned_config], axis=0)

        df = pd.concat([df, tuned_config], axis=1).fillna(method="ffill")
        expt_df = pd.concat([expt_df, df], axis=0)

    parameter_table = parameter_table.drop_duplicates()
    parameter_table.index = range(0, len(parameter_table.index))
    results_cv = expt_df.groupby(by=tuned_params + ["timesteps_total"]).mean()
    return results_cv


def get_best_expt(analysis: ExperimentAnalysis):
    results_cv = get_expt_results_cv(analysis)
    tuned_params = list(get_tuning_params(analysis.best_config["_algo"]).keys())
    last_5_avg: pd.DataFrame = (
        results_cv["evaluation/episode_reward_mean"].rolling(5).mean().groupby(tuned_params).last().sort_values(ascending=False)
    )
    best_config = dict(zip(last_5_avg.index.names, last_5_avg.index[0]))
    best_score = results_cv.loc[tuple(best_config.values()), :]
    return best_score, best_config


def expt_all_progress_cv(analysis: ExperimentAnalysis):
    expt_results_cv = get_expt_results_cv(analysis)
    tuned_params = list(get_tuning_params(analysis.best_config["_algo"]).keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for name, group in expt_results_cv.groupby(tuned_params):
        x = group.index.get_level_values(-1)
        reward_mean_train = group["episode_reward_mean"].rolling(5).mean().values
        reward_std_train = group["episode_reward_mean"].rolling(5).std().values
        reward_mean_eval = group["evaluation/episode_reward_mean"].rolling(5).mean().values
        reward_std_eval = group["evaluation/episode_reward_mean"].rolling(5).std().values
        axes[0].plot(x, reward_mean_train, label=name)
        axes[0].fill_between(
            x,
            y1=reward_mean_train + reward_std_train,
            y2=reward_mean_train - reward_std_train,
            alpha=0.2,
        )
        axes[1].plot(x, reward_mean_eval)
        axes[1].fill_between(
            x,
            y1=reward_mean_eval + reward_std_eval,
            y2=reward_mean_eval - reward_std_eval,
            alpha=0.2,
        )

        # axes[0].set_label(name)
        # axes[1].set_label(name)

    axes[0].grid()
    axes[1].grid()
    # axes[0].legend()
    # axes[1].legend()


def get_best_trials(analysis: ExperimentAnalysis, best_config: dict):
    all_configs = analysis.get_all_configs()
    best_trial_ids = []
    for logdir, df in analysis.trial_dataframes.items():
        config = all_configs[logdir].copy()
        is_match = True
        for key, value in best_config.items():
            if key == "env_config/window_size":
                continue
            if config[key] != value:
                is_match = False

        if is_match:
            best_trial_ids.append(df["trial_id"][0])

    best_trials = []
    for trial in analysis.trials:
        if trial.trial_id in best_trial_ids:
            best_trials.append(trial)

    print("Best Trials: ", best_trials)
    return best_trials


def backtest_expt(analysis: ExperimentAnalysis, debug=False):
    stats = pd.DataFrame()
    all_config = analysis.get_all_configs()
    best_score, best_config = get_best_expt(analysis)
    best_trials = get_best_trials(analysis, best_config)
    agent = None
    for trial in best_trials:
        config = all_config[trial.logdir].copy()
        agent_class, algo_config = prepare_config_for_agent(config, pathlib.Path(trial.logdir))
        env_test_config = algo_config.pop("_env_test_config")
        algo_config["num_workers"] = 1
        algo_config["logger_config"] = {"type": ray.tune.logger.NoopLogger}

        if agent is None:
            agent = agent_class(config=algo_config)
        else:
            agent.setup(algo_config)

        checkpoint = analysis.get_best_checkpoint(trial)
        agent.restore(checkpoint)

        env_train = TradingEnv(**algo_config["env_config"])
        env_eval = TradingEnv(**algo_config["evaluation_config"]["env_config"])
        env_test = TradingEnv(**env_test_config)
        stats_train = backtest(
            env_train,
            agent,
            save_dir=os.path.join(trial.logdir, "last-stats-train"),
            plot=True,
            open_browser=debug,
        )
        stats_eval = backtest(
            env_eval,
            agent,
            save_dir=os.path.join(trial.logdir, "last-stats-eval"),
            plot=True,
            open_browser=debug,
        )
        stats_test = backtest(
            env_test,
            agent,
            save_dir=os.path.join(trial.logdir, "last-stats-test"),
            plot=True,
        )
        stats_bh = backtest(env_test, agent="Buy&Hold", save_dir=os.path.join(trial.logdir, "buy-and-hold"), plot=False)

        stats = pd.concat([stats_train, stats_eval, stats_test, stats_bh], axis=1)
        stats.columns = ["train", "eval", "test", "buy&hold"]
        print(stats)

    return stats
