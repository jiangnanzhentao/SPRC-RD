from __future__ import annotations

import os
from types import SimpleNamespace
import gc
import torch
import torch.distributed as dist

from sprc_rd.training.objectives import (
    _all_finite,
    _image_aug_count,
    _image_aug_rng,
    _pseudo_count,
    choose_hard_same_image_shifts,
    corrupt_pseudo_subset_inplace,
    counterfactual_gain_loss,
    counterfactual_kl_loss,
    freeze_batchnorm_stats,
    image_aug_schedule,
    image_recovery_cosine_loss,
    make_large_missing_images,
    mixed_original_rd_loss,
)
from trainer import TRAINER
from trainer.rd_trainer import ReverseDistillationTrainer
from util.net import reduce_tensor, trans_state_dict
from util.util import log_msg, update_log_term


def _unwrap(net):
    return net.module if hasattr(net, "module") else net


@TRAINER.register_module
class SPRCRDTrainer(ReverseDistillationTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.loss_scaler is not None:
            raise RuntimeError(
                "The SPRC-RD schedule uses two ordered backward passes. "
                "Set cfg.trainer.scaler='none' (the supplied config already does)."
            )
        self._relation_epoch = 0
        ader_cal_anomaly_map = self.evaluator.cal_anomaly_map

        def sprc_rd_cal_anomaly_map(ft_list, fs_list, out_size, uni_am=False,
                                 use_cos=True, amap_mode='add', gaussian_sigma=0,
                                 weights=None):
            anomaly_map, layer_maps = ader_cal_anomaly_map(
                ft_list, fs_list, out_size, uni_am=uni_am, use_cos=use_cos,
                amap_mode=amap_mode, gaussian_sigma=gaussian_sigma, weights=weights,
            )
            if not uni_am and amap_mode == 'add':
                resolved_weights = weights if weights else [1] * len(ft_list)
                anomaly_map = anomaly_map * (len(ft_list) * sum(resolved_weights))
            return anomaly_map, layer_maps

        self.evaluator.cal_anomaly_map = sprc_rd_cal_anomaly_map
        report = _unwrap(self.net).parameter_report()
        if self.master:
            log_msg(self.logger, "==> SPRC-RD parameter report: {}".format(report))
            adapter = getattr(_unwrap(self.net), "teacher_adapter_report", None)
            log_msg(self.logger, "==> SPRC-RD teacher adapter: {}".format(adapter))

    def _method_cfg(self, name):
        return getattr(self.cfg.sprc_rd, name)

    def _begin_relation_epoch_if_needed(self, model):
        current = int(self.epoch) + 1
        if self._relation_epoch != current:
            model.bottleneck.begin_epoch(current)
            self._relation_epoch = current

    def _finalize_relation_epoch_if_needed(self, model):
        if self.iter % self.cfg.data.train_size != 0:
            return
        if dist.is_available() and dist.is_initialized():
            for scale in model.bottleneck.scales:
                dist.all_reduce(scale.relation_counts, op=dist.ReduceOp.SUM)
        model.bottleneck.finalize_relation_epoch()

    def optimize_parameters(self):
        model = _unwrap(self.net)
        self._begin_relation_epoch_if_needed(model)
        self.optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher = [feature.detach() for feature in model.net_t(self.imgs)]

        batch = teacher[0].shape[0]
        n_pseudo = _pseudo_count(batch, self._method_cfg("pseudo_fraction"))
        n_clean = batch - n_pseudo
        clean_pseudo_targets = None
        clean_coarse_appearance = None
        coarse_mask = None

        progress, image_weight, area_min, area_max = image_aug_schedule(
            int(self.epoch) + 1, self.cfg.sprc_rd
        )
        del progress
        image_targets = None
        corrupted_images = None
        image_mask = None
        n_image = 0

        if image_weight > 0.0 and n_clean > 0:
            n_image = _image_aug_count(
                batch, self._method_cfg("img_aug_fraction"), n_clean
            )
            if n_image > 0:
                batch_index = (int(self.iter) - 1) % int(self.cfg.data.train_size)
                rng = _image_aug_rng(
                    self.cfg.seed, int(self.epoch) + 1, batch_index
                )
                chosen = sorted(rng.sample(range(n_clean), n_image))
                indices = torch.tensor(chosen, device=self.imgs.device, dtype=torch.long)
                image_targets = [
                    feature.index_select(0, indices).detach().clone() for feature in teacher
                ]
                source_images = self.imgs.index_select(0, indices)
                corrupted_images, image_mask = make_large_missing_images(
                    source_images,
                    min_fraction=area_min,
                    max_fraction=area_max,
                    rng=rng,
                    mean_fill_probability=self._method_cfg("img_aug_mean_fill_probability"),
                )

        if n_pseudo > 0:
            clean_pseudo_targets = [feature[n_clean:].clone() for feature in teacher]
            with torch.no_grad():
                clean_coarse_appearance = model.bottleneck.extract_clean_coarse_appearance(
                    clean_pseudo_targets[2]
                )
                height, width = clean_pseudo_targets[2].shape[-2:]
                from sprc_rd.training.objectives import generate_coarse_mask
                coarse_mask = generate_coarse_mask(
                    n_pseudo,
                    height,
                    width,
                    device=self.imgs.device,
                    min_blocks=self._method_cfg("mask_min_blocks"),
                    max_blocks=self._method_cfg("mask_max_blocks"),
                    min_side=self._method_cfg("mask_min_side"),
                    max_side=self._method_cfg("mask_max_side"),
                    stripe_prob=self._method_cfg("mask_stripe_prob"),
                    margin=self._method_cfg("mask_margin"),
                    max_fraction=self._method_cfg("mask_max_fraction"),
                )
                shifts, _ = choose_hard_same_image_shifts(
                    clean_pseudo_targets[2],
                    coarse_mask,
                    min_shift=self._method_cfg("donor_min_shift"),
                    max_shift=self._method_cfg("donor_max_shift"),
                    candidates=self._method_cfg("donor_candidates"),
                    max_source_overlap=self._method_cfg("max_source_overlap"),
                )
                corrupt_pseudo_subset_inplace(
                    teacher,
                    clean_pseudo_targets,
                    clean_count=n_clean,
                    coarse_mask=coarse_mask,
                    shifts=shifts,
                    missing_probability=self._method_cfg("pseudo_missing_probability"),
                    missing_kernel=self._method_cfg("pseudo_missing_kernel"),
                )

        with self.amp_autocast():
            
            
            teacher, student, aux = self.net(
                None, teacher_override=teacher, return_aux=True
            )
            loss_rd, _, _, _ = mixed_original_rd_loss(
                teacher,
                clean_pseudo_targets,
                student,
                clean_count=n_clean,
                pseudo_weight=self._method_cfg("pseudo_rd_weight"),
            )
            if float(model.bottleneck.scales[2].relation_alpha) > 0.0:
                loss_cf = counterfactual_kl_loss(
                    aux["coarse"]["cf_q"],
                    clean_coarse_appearance,
                    coarse_mask,
                    clean_count=n_clean,
                )
                loss_gain, _, _ = counterfactual_gain_loss(
                    aux["multiscale_structural_gain"],
                    coarse_mask,
                    clean_count=n_clean,
                    pseudo_margin=self._method_cfg("train_gain_margin"),
                    clean_tolerance=self._method_cfg("clean_gain_tolerance"),
                )
            else:
                loss_cf = loss_rd.new_tensor(0.0)
                loss_gain = loss_rd.new_tensor(0.0)
            loss_main = (
                loss_rd
                + float(self._method_cfg("lambda_cf")) * loss_cf
                + float(self._method_cfg("lambda_gain")) * loss_gain
            )

        if not bool(torch.isfinite(loss_main).item()):
            raise RuntimeError("non-finite SPRC-RD training loss")
        loss_main.backward()

        loss_image = loss_main.new_tensor(0.0)
        if n_image > 0 and corrupted_images is not None:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                corrupted_features = model.net_t(corrupted_images.float())
            finite = _all_finite(corrupted_features)
            if finite:
                bn_context = (
                    torch.enable_grad()
                    if self._method_cfg("img_aug_update_bn_stats")
                    else freeze_batchnorm_stats(model.bottleneck, model.net_s)
                )
                with bn_context, torch.cuda.amp.autocast(enabled=False):
                    _, image_prediction = self.net(
                        None,
                        teacher_override=[feature.float() for feature in corrupted_features],
                        return_aux=False,
                    )
                    finite = _all_finite(list(image_prediction))
            if finite:
                with torch.cuda.amp.autocast(enabled=False):
                    loss_global = image_recovery_cosine_loss(
                        image_targets, image_prediction, image_mask=None
                    )
                    loss_masked = image_recovery_cosine_loss(
                        image_targets, image_prediction, image_mask=image_mask
                    )
                    loss_image = loss_global + float(
                        self._method_cfg("img_aug_masked_weight")
                    ) * loss_masked
                finite = bool(torch.isfinite(loss_image).item())
            if finite:
                (float(image_weight) * loss_image).backward()

        if self.cfg.loss.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                list(model.bottleneck.parameters()) + list(model.net_s.parameters()),
                max_norm=float(self.cfg.loss.clip_grad),
            )
        self.optim.step()
        model.bottleneck.accumulate_relation_counts(aux, teacher, clean_count=n_clean)
        self._finalize_relation_epoch_if_needed(model)

        total = loss_main.detach() + float(image_weight) * loss_image.detach()
        values = {
            "total": total,
            "rd": loss_rd,
            "cf": loss_cf,
            "gain": loss_gain,
            "img": loss_image,
        }
        for name, value in values.items():
            update_log_term(
                self.log_terms.get(name),
                reduce_tensor(value, self.world_size).detach().item(),
                1,
                self.master,
            )

    def _deployment_state(self, model):
        architecture = dict(self.cfg.model.kwargs)
        architecture.pop("model_t", None)
        architecture.pop("model_s", None)
        architecture.pop("model_checkpoint_path", None)
        return {
            "epoch": int(self.epoch_full),
            "backbone": model.backbone,
            "outer_impl": "ader",
            "ader_teacher_state": trans_state_dict(model.net_t.state_dict(), dist=False),
            "bottleneck": trans_state_dict(model.bottleneck.state_dict(), dist=False),
            "decoder": trans_state_dict(model.net_s.state_dict(), dist=False),
            "args": architecture,
            "method": "SPRC-RD",
            "modules": {
                "projection": "Multi-scale Structural Prototype Projection",
                "intervention": "Relation-Guided Prototype Intervention",
                "calibration": "Prototype-State Residual Calibration",
            },
        }

    def _finish(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.master:
            model = _unwrap(self.net)
            final_path = os.path.join(self.cfg.logdir, "sprc_rd_final.pth")
            torch.save(self._deployment_state(model), final_path)
            log_msg(self.logger, "==> Saved epoch-200 SPRC-RD model: {}".format(final_path))
            from sprc_rd.training.objectives import (
                postfit_state_residual_checkpoint,
            )

            postfit_args = SimpleNamespace(
                data_root=self.cfg.data.root,
                class_name="all",
                class_list=",".join(str(name) for name in self.cls_names),
                train_normal_folder="good",
                no_recursive_train=False,
                state_residual_batch_size=int(
                    self._method_cfg("state_residual_batch_size")
                ),
                state_residual_num_workers=int(
                    self._method_cfg("state_residual_num_workers")
                ),
                state_residual_export_npz=bool(
                    self._method_cfg("state_residual_export_npz")
                ),
                num_workers=self.cfg.trainer.data.num_workers_per_gpu,
                seed=self.cfg.seed,
                resize=self.cfg.size,
                input_size=self.cfg.size,
                outer_impl="ader",
                teacher_checkpoint=str(
                    self.cfg.model.kwargs.get("teacher_checkpoint", "")
                ),
                cpu=False,
                no_deterministic=not bool(self.cfg.trainer.cuda_deterministic),
                allow_tf32=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if hasattr(self, "optim") and self.optim is not None:
                self.optim.zero_grad(set_to_none=True)

            for attr_name in (
                    "feats_t",
                    "feats_s",
                    "imgs",
                    "imgs_mask",
                    "anomaly_map",
            ):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)

            for attr_name in (
                    "optim",
                    "scheduler",
                    "loss_scaler",
                    "net",
            ):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except RuntimeError:
                    pass

            postfit_state_residual_checkpoint(final_path, postfit_args)
            log_msg(
                self.logger,
                "==> Calibration embedded in final model: {}".format(final_path),
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return super()._finish()
