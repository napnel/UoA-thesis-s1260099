def plot_best_progress_cv(analysis: ExperimentAnalysis):
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