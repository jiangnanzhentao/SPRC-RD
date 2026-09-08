import logging
import os
import shutil
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter


def init_checkpoint(cfg):
    os.makedirs(cfg.trainer.checkpoint, exist_ok=True)
    if cfg.trainer.resume_dir:
        cfg.logdir = os.path.join(cfg.trainer.checkpoint, cfg.trainer.resume_dir)
        cfg.model.kwargs["checkpoint_path"] = os.path.join(
            cfg.logdir, "training_state.pth"
        )
        state = torch.load(
            cfg.model.kwargs["checkpoint_path"], map_location="cpu", weights_only=False
        )
        cfg.trainer.iter = int(state["iter"])
        cfg.trainer.epoch = int(state["epoch"])
    else:
        if cfg.master:
            suffix = cfg.trainer.logdir_sub or time.strftime("%Y%m%d-%H%M%S")
            base = "SPRC-RD_{}".format(suffix)
            cfg.logdir = os.path.join(cfg.trainer.checkpoint, base)
            index = 1
            while os.path.exists(cfg.logdir):
                cfg.logdir = os.path.join(cfg.trainer.checkpoint, "{}_{}".format(base, index))
                index += 1
            os.makedirs(cfg.logdir)
            source = cfg.cfg_path.removesuffix(".py") + ".py"
            shutil.copy2(source, os.path.join(cfg.logdir, os.path.basename(source)))
        else:
            cfg.logdir = None
        cfg.trainer.iter = 0
        cfg.trainer.epoch = 0
    cfg.logger = get_logger(cfg) if cfg.master else None
    cfg.writer = SummaryWriter(cfg.logdir) if cfg.master else None


def get_logger(cfg):
    logger = logging.getLogger("SPRC-RD-rank-{}".format(cfg.rank))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%m/%d %I:%M:%S %p"
        )
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(
            os.path.join(cfg.logdir, "log_{}.txt".format(cfg.mode))
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    return logger


def log_cfg(cfg):
    flattened = {}

    def visit(value, prefix=""):
        if hasattr(value, "__dict__"):
            for key, item in vars(value).items():
                if not key.startswith("_") and key not in ("logger", "writer"):
                    visit(item, "{}.{}".format(prefix, key).lstrip("."))
        else:
            flattened[prefix] = value

    visit(cfg)
    width = max(map(len, flattened))
    cfg.cfg_dict = flattened
    cfg.cfg_str = "\n".join(
        "{:<{}} : {}".format(key, width, value) for key, value in flattened.items()
    )
    log_msg(cfg.logger, "==> configuration\n{}".format(cfg.cfg_str))


def log_msg(logger, message, level="info"):
    if logger is not None and message is not None:
        getattr(logger, level)(message)


def able(value, condition=False, default=None):
    return value if condition else default


class AvgMeter:
    def __init__(self, name, fmt=":f", show_name="val", add_name=""):
        self.name = name
        self.fmt = fmt
        self.show_name = show_name
        self.add_name = add_name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, count=1):
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count

    def __str__(self):
        text = "[{name} {" + self.show_name + self.fmt + "}"
        if self.add_name:
            text += " ({" + self.add_name + self.fmt + "})"
        return (text + "]").format(**vars(self))


class ProgressMeter:
    def __init__(self, meters, default_prefix=""):
        self.meters = meters
        self.default_prefix = default_prefix

    def get_msg(self, iteration, total, epoch=None, epochs=None, prefix=None):
        label = prefix or self.default_prefix
        entries = ["{}: {:>6.2f}% [{}/{}]".format(label, iteration / total * 100, iteration, total)]
        if epoch is not None and epochs is not None:
            entries.append("[{:.1f}/{:.1f}]".format(epoch, epochs))
        entries.extend(str(meter) for meter in self.meters.values() if meter.count)
        return " ".join(entries)


def get_log_terms(configs, default_prefix=""):
    terms = {}
    for config in configs:
        kwargs = dict(config)
        name = kwargs["name"]
        suffixes = kwargs.pop("suffixes", None)
        if suffixes is None:
            terms[name] = AvgMeter(**kwargs)
        else:
            for suffix in suffixes:
                item = dict(kwargs)
                item["name"] = name + suffix
                terms[item["name"]] = AvgMeter(**item)
    return terms, ProgressMeter(terms, default_prefix)


def update_log_term(term, value, count, master):
    if term is not None and master:
        term.update(value, count)
