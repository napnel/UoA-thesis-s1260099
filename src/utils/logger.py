import os
from argparse import Namespace
import tempfile
from datetime import datetime

from ray.tune.logger import UnifiedLogger


def create_log_filename(args: Namespace):
    filename = ""

    omited_list = ("backtest", "stop_timesteps", "checkpoint")

    for key, value in args.__dict__.items():
        if value and key not in omited_list:
            if isinstance(value, list):
                value = tuple(value)
            filename += f"({key}={value})_"

    filename = filename.replace(" ", "")
    filename = filename[:-1]
    return filename


def custom_log_creator(custom_path: str, custom_str: str = ""):
    custom_path = os.path.expanduser(custom_path)

    timestr = datetime.today().strftime("%m-%d_%H-%M")
    logdir_prefix = f"{custom_str}_{timestr}"

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        # logdir = os.path.join(custom_path, logdir_prefix)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator
