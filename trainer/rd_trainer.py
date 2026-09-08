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
        model = self.net.module if hasattr(self.net, "module") else self.net
        if model.residual_calibrator is None:
            raise RuntimeError(
                "The loaded checkpoint does not contain PSRC calibration statistics. "
                "Use sprc_rd_final.pth generated after epoch 200."
            )
        local = dict(
            imgs_masks=[],
            anomaly_maps=[],
            image_scores=[],
            cls_names=[],
            anomalys=[],
        )
        for batch_index, test_data in enumerate(self.test_loader, 1):
            started = get_timepc()
            self.set_input(test_data)
            self.feats_t, self.feats_s, aux = model.forward_features(
                self.imgs, return_aux=True
            )
            loss_cos = self.loss_terms["cos"](self.feats_t, self.feats_s)
            update_log_term(
                self.log_terms.get("cos"),
                reduce_tensor(loss_cos, self.world_size).item(),
                1,
                self.master,
            )
            raw_map, _ = self.evaluator.cal_anomaly_map(
                self.feats_t,
                self.feats_s,
                self.imgs.shape[-2:],
                uni_am=False,
                amap_mode="add",
                gaussian_sigma=4,
            )
            calibrated_map = model.residual_calibrator(
                self.feats_t,
                self.feats_s,
                aux,
                out_hw=tuple(self.imgs.shape[-2:]),
            )
            calibrated_map = calibrated_map.detach().float().cpu().numpy()
            if calibrated_map.ndim == 4 and calibrated_map.shape[1] == 1:
                calibrated_map = calibrated_map[:, 0]
            image_scores = raw_map.reshape(raw_map.shape[0], -1).max(axis=1)
            masks = (self.imgs_mask > 0.5).cpu().numpy().astype(int)
            local["imgs_masks"].append(masks)
            local["anomaly_maps"].append(calibrated_map)
            local["image_scores"].append(image_scores)
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
