import datetime
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from data import get_loader
from loss import get_loss_terms
from model import get_model
from optim import get_optim
from optim.scheduler import get_scheduler
from util.metric import get_evaluator
from util.net import get_autocast, get_timepc, trans_state_dict
from util.util import able, get_log_terms, log_cfg, log_msg, update_log_term


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.master = cfg.master
        self.logger = cfg.logger
        self.writer = cfg.writer
        self.rank = cfg.rank
        self.world_size = cfg.world_size
        self.device = torch.device("cuda", cfg.local_rank)
        self.net = get_model(cfg.model).to(self.device)
        self.optim = get_optim(cfg.optim.kwargs, self.net, lr=cfg.optim.lr)
        self.amp_autocast = get_autocast(cfg.trainer.scaler)
        self.loss_scaler = None
        self.loss_terms = get_loss_terms(cfg.loss.loss_terms, device=self.device)
        if cfg.dist:
            self.net = DistributedDataParallel(
                self.net,
                device_ids=[cfg.local_rank],
                find_unused_parameters=cfg.trainer.find_unused_parameters,
            )
        self.train_loader, self.test_loader = get_loader(cfg)
        cfg.data.train_size = len(self.train_loader)
        cfg.data.test_size = len(self.test_loader)
        cfg.data.train_length = len(self.train_loader.dataset)
        cfg.data.test_length = len(self.test_loader.dataset)
        self.cls_names = self.train_loader.dataset.cls_names
        self.scheduler = get_scheduler(cfg, self.optim)
        self.evaluator = get_evaluator(cfg.evaluator)
        self.metrics = self.evaluator.metrics
        self.iter = int(cfg.trainer.iter)
        self.epoch = int(cfg.trainer.epoch)
        self.iter_full = int(cfg.trainer.iter_full)
        self.epoch_full = int(cfg.trainer.epoch_full)
        if cfg.trainer.resume_dir:
            state = torch.load(
                cfg.model.kwargs["checkpoint_path"], map_location="cpu", weights_only=False
            )
            self.optim.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            cfg.task_start_time = get_timepc() - float(state["total_time"])
        log_cfg(cfg)

    def reset(self, is_train=True):
        self.net.train(is_train)
        terms = able(
            self.cfg.logging.log_terms_train,
            is_train,
            self.cfg.logging.log_terms_test,
        )
        self.log_terms, self.progress = get_log_terms(
            terms, default_prefix="Train" if is_train else "Test"
        )

    def scheduler_step(self):
        self.scheduler.step(self.iter)
        update_log_term(
            self.log_terms.get("lr"), self.optim.param_groups[0]["lr"], 1, self.master
        )

    def set_input(self, inputs):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError

    def train(self):
        self.reset(True)
        train_length = self.cfg.data.train_size
        while self.epoch < self.epoch_full:
            if self.cfg.dist:
                self.train_loader.sampler.set_epoch(self.epoch)
            for train_data in self.train_loader:
                if self.iter >= self.iter_full:
                    break
                self.scheduler_step()
                started = get_timepc()
                self.iter += 1
                self.set_input(train_data)
                loaded = get_timepc()
                self.optimize_parameters()
                finished = get_timepc()
                update_log_term(self.log_terms.get("data_t"), loaded - started, 1, self.master)
                update_log_term(self.log_terms.get("optim_t"), finished - loaded, 1, self.master)
                update_log_term(self.log_terms.get("batch_t"), finished - started, 1, self.master)
                if self.master and self.iter % self.cfg.logging.train_log_per == 0:
                    log_msg(
                        self.logger,
                        self.progress.get_msg(
                            self.iter,
                            self.iter_full,
                            self.iter / train_length,
                            self.epoch_full,
                        ),
                    )
                    if self.writer:
                        for name, meter in self.log_terms.items():
                            self.writer.add_scalar("Train/{}".format(name), meter.val, self.iter)
                        self.writer.flush()
                if self.iter % self.cfg.logging.train_reset_log_per == 0:
                    self.reset(True)
            self.epoch += 1
            self.cfg.total_time = get_timepc() - self.cfg.task_start_time
            elapsed = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
            remaining = str(
                datetime.timedelta(
                    seconds=int(
                        self.cfg.total_time
                        / max(self.epoch, 1)
                        * (self.epoch_full - self.epoch)
                    )
                )
            )
            log_msg(
                self.logger,
                "==> epoch {}/{} | elapsed {} | remaining {}".format(
                    self.epoch, self.epoch_full, elapsed, remaining
                ),
            )
            self.save_training_state()
            self.reset(True)
        self._finish()

    def save_training_state(self):
        if not self.master:
            return
        state = {
            "net": trans_state_dict(self.net.state_dict(), dist=False),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "iter": self.iter,
            "epoch": self.epoch,
            "total_time": self.cfg.total_time,
        }
        torch.save(state, os.path.join(self.cfg.logdir, "training_state.pth"))

    def test(self):
        raise NotImplementedError

    def _finish(self):
        log_msg(self.logger, "==> SPRC-RD training and calibration complete")
        if self.master and self.writer:
            self.writer.close()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def run(self):
        if self.cfg.mode == "train":
            self.train()
        elif self.cfg.mode == "test":
            self.test()
        else:
            raise ValueError("unsupported mode: {}".format(self.cfg.mode))
