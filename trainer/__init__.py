from util.registry import Registry


TRAINER = Registry("Trainer")


def get_trainer(cfg):
    return TRAINER.get_module(cfg.trainer.name)(cfg)


from . import sprc_rd_trainer
