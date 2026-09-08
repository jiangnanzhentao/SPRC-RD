import importlib
from argparse import Namespace
from ast import literal_eval

from util.net import get_timepc


def get_cfg(terminal):
    module_path = terminal.cfg_path.removesuffix(".py").replace("/", ".")
    cfg = importlib.import_module(module_path).cfg()
    for key, value in vars(terminal).items():
        setattr(cfg, key, value)
    for option in cfg.opts:
        path, raw_value = option.split("=", 1)
        try:
            value = literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        target = cfg
        keys = path.split(".")
        for key in keys[:-1]:
            if isinstance(target, dict):
                target = target.setdefault(key, {})
            else:
                if not hasattr(target, key):
                    setattr(target, key, Namespace())
                target = getattr(target, key)
        if isinstance(target, dict):
            target[keys[-1]] = value
        else:
            setattr(target, keys[-1], value)
    cfg.task_start_time = get_timepc()
    return cfg
