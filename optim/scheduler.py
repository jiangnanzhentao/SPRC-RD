import torch


class IterationMultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(self, optimizer, milestones, gamma, iterations_per_epoch):
        steps = [int(epoch) * int(iterations_per_epoch) for epoch in milestones]
        super().__init__(optimizer, milestones=steps, gamma=float(gamma))


def get_scheduler(cfg, optimizer):
    kwargs = dict(cfg.trainer.scheduler_kwargs)
    if kwargs.pop("name") != "multistep":
        raise ValueError("SPRC-RD uses the multistep scheduler")
    kwargs.pop("use_iters", None)
    cfg.trainer.iter_full = cfg.data.train_size * cfg.trainer.epoch_full
    return IterationMultiStepLR(
        optimizer,
        milestones=kwargs.get("milestones", [100, 120]),
        gamma=kwargs.get("gamma", 0.2),
        iterations_per_epoch=cfg.data.train_size,
    )
