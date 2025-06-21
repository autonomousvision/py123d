import os

import easydict
import yaml


def load_config_act(path):
    """load config file"""
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return easydict.EasyDict(cfg)


def load_config_init(path):
    """load config file"""
    path = os.path.join("init/configs", f"{path}.yaml")
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
