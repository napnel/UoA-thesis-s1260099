from typing import Dict, Any
from ray.rllib.agents.trainer import Trainer
from envs.trading_env import ContTradingEnv, DescTradingEnv


class Trainer:
    @classmethod
    def learn(self, agent: Trainer, timesteps_total=1e6, episode_reward_mean=None, checkpoint_freq=10, verbose=1):
        checkpoint_path = f"./ray_results/{agent.__class__.__name__}"
        max_episode_reward_mean = 0
        i = 0
        while True:
            result = agent.train()
            # stop training of the target train steps or reward are reached
            # if result["timesteps_total"] >= args.stop_timesteps or result["episode_reward_mean"] >= args.stop_reward:
            #     break

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
    def get_agent_from_str(self, algo: str, user_config: Dict[str, Any]) -> Trainer:
        if algo == "DQN":
            from ray.rllib.agents import dqn

            config = dqn.DEFAULT_CONFIG.copy()
            config.update(user_config)
            agent = dqn.DQNTrainer(config=config)

        elif algo == "A2C":
            from ray.rllib.agents import a3c

            config = a3c.DEFAULT_CONFIG.copy()
            config.update(user_config)
            agent = a3c.A2CTrainer(config=config)

        elif algo == "PPO":
            from ray.rllib.agents import ppo

            config = ppo.DEFAULT_CONFIG.copy()
            config.update(user_config)
            agent = ppo.PPOTrainer(config=config)

        elif algo == "SAC":
            from ray.rllib.agents import sac

            config = sac.DEFAULT_CONFIG.copy()
            config.update(user_config)
            agent = sac.SACTrainer(config=config)

        return agent

    @classmethod
    def get_env_from_str(self, env_name: str, env_config={}) -> Trainer:
        if env_name == "DescTradingEnv":
            env = DescTradingEnv(**env_config)

        elif env_name == "ContTradingEnv":
            env = ContTradingEnv(**env_config)

        else:
            raise ValueError

        return env
