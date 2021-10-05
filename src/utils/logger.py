import os
import tempfile
from datetime import datetime

from ray.tune.logger import UnifiedLogger


def custom_log_creator(custom_path, custom_str):
    custom_path = os.path.expanduser(custom_path)

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = f"{custom_str}_{timestr}"

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

