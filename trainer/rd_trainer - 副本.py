import numpy as np
import tabulate
import torch
import torch.distributed as dist

from ._base_trainer import BaseTrainer
from util.net import get_timepc, reduce_tensor
from util.util import log_msg, update_log_term


class ReverseDistillationTrainer(BaseTrainer):
    def set_input(self, inputs):
        self.imgs = inputs["img"].to(self.device, non_blocking=True)
        self.imgs_mask = inputs["img_mask"].to(self.device, non_blocking=True)
        self.cls_name = inputs["cls_name"]
        self.anomaly = inputs["anomaly"]
        self.img_path = inputs["img_path"]

    def forward(self):
        self.feats_t, self.feats_s = self.net(self.imgs)

    @torch.no_grad()
    def test(self):
        self.reset(False)
        local = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
        for batch_index, test_data in enumerate(self.test_loader, 1):
            started = get_timepc()
            self.set_input(test_data)
            self.forward()
            loss_cos = self.loss_terms["cos"](self.feats_t, self.feats_s)
            update_log_term(
                self.log_terms.get("cos"),
                reduce_tensor(loss_cos, self.world_size).item(),
                1,
                self.master,
            )
            anomaly_map, _ = self.evaluator.cal_anomaly_map(
                self.feats_t,
                self.feats_s,
                self.imgs.shape[-2:],
                uni_am=False,
                amap_mode="add",
                gaussian_sigma=4,
            )
            masks = (self.imgs_mask > 0.5).cpu().numpy().astype(int)
            local["imgs_masks"].append(masks)
            local["anomaly_maps"].append(anomaly_map)
            local["cls_names"].append(np.asarray(self.cls_name))
            local["anomalys"].append(self.anomaly.cpu().numpy().astype(int))
            update_log_term(
                self.log_terms.get("batch_t"), get_timepc() - started, 1, self.master
            )
            if self.master and (
                batch_index % self.cfg.logging.test_log_per == 0
                or batch_index == len(self.test_loader)
            ):
                log_msg(
                    self.logger,
                    self.progress.get_msg(batch_index, len(self.test_loader), prefix="Test"),
                )
        gathered = [None] * self.world_size if self.cfg.dist else [local]
        if self.cfg.dist:
            dist.all_gather_object(gathered, local)
        if not self.master:
            return None
        merged = {name: [] for name in local}
        for part in gathered:
            for name, values in part.items():
                merged[name].extend(values)
        results = {name: np.concatenate(values, axis=0) for name, values in merged.items()}
        report = {"Name": []}
        per_metric = {metric: [] for metric in self.metrics}
        for cls_name in self.cls_names:
            values = self.evaluator.run(results, cls_name, self.logger)
            report["Name"].append(cls_name)
            for metric in self.metrics:
                value = float(values[metric]) * 100
                report.setdefault(metric, []).append(value)
                per_metric[metric].append(value)
        if len(self.cls_names) > 1:
            report["Name"].append("Avg")
            for metric in self.metrics:
                report[metric].append(float(np.mean(per_metric[metric])))
        table = tabulate.tabulate(
            report,
            headers="keys",
            tablefmt="pipe",
            floatfmt=".3f",
            numalign="center",
            stralign="center",
        )
        log_msg(self.logger, "\n{}".format(table))
        return report
