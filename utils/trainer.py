from tabnanny import check
from typing import Dict, Any, Tuple
from ray.rllib.agents.trainer import Trainer
from envs.trading_env import ContTradingEnv, DescTradingEnv


class Trainer:
    @classmethod
    def learn(self, agent: Trainer, timesteps_total=1e6, episode_reward_mean=None, checkpoint_freq=10, verbose=1) -> Tuple[Trainer, str]:
        checkpoint_path = f"./ray_results/{agent.__class__.__name__}"
        max_episode_reward_mean = 0
        i = 0
        while True:
            result = agent.train()
            # stop training of the target train steps or reward are reached
            # if result["timesteps_total"] >= args.stop_timesteps or result["episode_reward_mean"] >= args.stop_reward:
            #     break

            # if result["evaluation"]["episode_reward_mean"] > max_episode_reward_mean:
            #     print("episodes_total: ", result["episodes_total"], "timesptes_total: ", result["timesteps_total"])
            #     print("Train", result["episode_reward_mean"], "| Eval", result["evaluation"]["episode_reward_mean"])
            #     max_episode_reward_mean = result["evaluation"]["episode_reward_mean"]
            #     checkpoint = agent.save(checkpoint)

            if i > 0 and i % checkpoint_freq == 0:
                print("episodes_total: ", result["episodes_total"], "timesptes_total: ", result["timesteps_total"])
                print("Train", result["episode_reward_mean"], "| Eval", result["evaluation"]["episode_reward_mean"])
                checkpoint = agent.save(checkpoint_path)
                print("checkpoint saved at", checkpoint)

            if result["timesteps_total"] > timesteps_total:
                break

            i += 1

        checkpoint = agent.save(checkpoint_path)
        return agent, checkpoint

    @classmethod
    def get_agent_class(self, algo: str) -> Tuple[Trainer, Dict]:
        if algo == "DQN":
            from ray.rllib.agents import dqn

            return dqn.DQNTrainer, dqn.DEFAULT_CONFIG.copy()

        elif algo == "A2C":
            from ray.rllib.agents import a3c

            return a3c.A2CTrainer, a3c.DEFAULT_CONFIG.copy()

        elif algo == "PPO":
            from ray.rllib.agents import ppo

            return ppo.PPOTrainer, ppo.DEFAULT_CONFIG.copy()

        elif algo == "SAC":
            from ray.rllib.agents import sac

            return sac.SACTrainer, sac.DEFAULT_CONFIG.copy()

        else:
            raise ValueError

    @classmethod
    def get_env(self, env_name: str, env_config={}):
        if env_name == "DescTradingEnv":
            env = DescTradingEnv(**env_config)

        elif env_name == "ContTradingEnv":
            env = ContTradingEnv(**env_config)

        else:
            raise ValueError

        return env
