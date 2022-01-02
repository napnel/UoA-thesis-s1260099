import ray
from ray import tune


def get_tuning_params(algo: str):
    tuning_params = {
        "lr": tune.uniform(1e-5, 1e-2),
        "gamma": tune.uniform(0.8, 0.99),
    }

    if algo == "DQN":
        tuning_params["n_step"] = tune.randint(1, 10)

    if algo == "A2C":
        tuning_params["lambda"] = tune.uniform(0.9, 1)
        tuning_params["entropy_coeff"] = tune.uniform(0, 1)

    if algo == "PPO":
        tuning_params["lambda"] = tune.uniform(0.9, 1)
        tuning_params["entropy_coeff"] = tune.uniform(0, 1)

    if algo == "SAC":
        tuning_params["tau"] = tune.uniform(5e-4, 5e-2)
        tuning_params["n_step"] = tune.randint(1, 10)

    return tuning_params
