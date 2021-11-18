import pathlib
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.agents import dqn, a3c, ppo, sac, ddpg
from src.utils import DataLoader, Preprocessor
from src.utils.misc import prepare_config_for_agent


class ExperimentCV(tune.Trainable):
    def setup(self, config: Dict[str, Any]):
        agent_class, algo_config = prepare_config_for_agent(config, self.logdir)
        self.agent = agent_class(config=algo_config)

    def step(self):
        return self.agent.train()

    def save_checkpoint(self, checkpoint_dir: str):
        return self.agent.save(pathlib.Path(checkpoint_dir).parent)
