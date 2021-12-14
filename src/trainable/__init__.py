from ray.tune import register_trainable
from src.trainable.cross_validation import ExperimentCV

register_trainable(ExperimentCV.__name__, ExperimentCV)
