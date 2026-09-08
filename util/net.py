import datetime
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


def init_training(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("SPRC-RD training requires CUDA")
    torch.cuda.empty_cache()
    cudnn.deterministic = bool(cfg.trainer.cuda_deterministic)
    cudnn.benchmark = not bool(cfg.trainer.cuda_deterministic)
    cfg.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cfg.rank = int(os.environ.get("RANK", "0"))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cfg.dist = cfg.world_size > 1
    cfg.ngpus_per_node = torch.cuda.device_count()
    cfg.nnodes = max(1, cfg.world_size // max(cfg.ngpus_per_node, 1))
    if cfg.dist:
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=cfg.dist_url,
            timeout=datetime.timedelta(hours=20),
        )
        dist.barrier()
    cfg.master = cfg.rank == cfg.logger_rank
    seed = cfg.seed + cfg.local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = cfg.trainer.data
    if data.batch_size:
        if data.batch_size % cfg.world_size:
            raise ValueError("batch size must be divisible by world size")
        data.batch_size_per_gpu = data.batch_size // cfg.world_size
    else:
        data.batch_size = data.batch_size_per_gpu * cfg.world_size
    if data.batch_size_test:
        if data.batch_size_test % cfg.world_size:
            raise ValueError("test batch size must be divisible by world size")
        data.batch_size_per_gpu_test = data.batch_size_test // cfg.world_size
    else:
        data.batch_size_test = data.batch_size_per_gpu_test * cfg.world_size


def trans_state_dict(state_dict, dist=False):
    output = {}
    for key, value in state_dict.items():
        if dist:
            key = key if key.startswith("module.") else "module." + key
        else:
            key = key[7:] if key.startswith("module.") else key
        output[key] = value
    return output


def get_timepc():
    return time.perf_counter()


def reduce_tensor(tensor, world_size):
    reduced = tensor.detach()
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced = reduced / world_size
    return reduced


def get_autocast(mode="none"):
    if mode == "none":
        return nullcontext
    if mode == "native":
        return torch.cuda.amp.autocast
    raise ValueError("unsupported scaler mode: {}".format(mode))
