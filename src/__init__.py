from ray.rllib.models import ModelCatalog
from ray.tune import register_trainable

from src.experiments import ExperimentCV
from src.models.batch_norm import BatchNormModel

ModelCatalog.register_custom_model("batch_norm_model", BatchNormModel)
register_trainable(ExperimentCV.__name__, ExperimentCV)
