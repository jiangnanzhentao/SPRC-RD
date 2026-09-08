import torch
from torch.utils.data.distributed import DistributedSampler

from util.registry import Registry


TRANSFORMS = Registry("Transforms")
DATA = Registry("Data")


from . import transforms
from .ad_dataset import DefaultAD
from .utils import get_transforms


def get_loader(cfg):
    train_transforms = get_transforms(cfg, True, cfg.data.train_transforms)
    test_transforms = get_transforms(cfg, False, cfg.data.test_transforms)
    target_transforms = get_transforms(cfg, False, cfg.data.target_transforms)
    dataset_type = DATA.get_module(cfg.data.type)
    train_set = dataset_type(cfg, True, train_transforms, target_transforms)
    test_set = dataset_type(cfg, False, test_transforms, target_transforms)
    train_sampler = DistributedSampler(train_set, shuffle=True) if cfg.dist else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if cfg.dist else None
    common = dict(
        num_workers=cfg.trainer.data.num_workers_per_gpu,
        pin_memory=cfg.trainer.data.pin_memory,
        persistent_workers=cfg.trainer.data.persistent_workers,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.trainer.data.batch_size_per_gpu,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=cfg.trainer.data.drop_last,
        **common,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.trainer.data.batch_size_per_gpu_test,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader
