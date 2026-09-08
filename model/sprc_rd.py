











from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Dict, Sequence

import torch
import torch.nn as nn

from sprc_rd.models.ader_encoder import create_ader_teacher
from sprc_rd.models.structural_prototype import StructuralPrototypeBottleneck
from sprc_rd.state_residual_calibration import PrototypeStateResidualCalibrator
from model import MODEL, get_model


BACKBONE_SPECS = {
    "wide_resnet50_2": ((256, 512, 1024), 2048),
    "resnet50": ((256, 512, 1024), 2048),
    "resnet34": ((64, 128, 256), 512),
    "resnet18": ((64, 128, 256), 512),
}


def parse_triplet(value, caster, name):
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    values = tuple(caster(item) for item in value)
    if len(values) != 3:
        raise ValueError("{} must contain three values".format(name))
    return values

_CANON_ROOTS = (
    "conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "fc."
)


def _load_checkpoint_file(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strict_load(module: nn.Module, state, name: str) -> None:
    try:
        module.load_state_dict(state, strict=True)
        return
    except RuntimeError as first_error:
        keys = list(state) if isinstance(state, Mapping) else []
        if keys and all(str(key).startswith("module.") for key in keys):
            stripped = {str(key)[7:]: value for key, value in state.items()}
            try:
                module.load_state_dict(stripped, strict=True)
                return
            except RuntimeError:
                pass
        raise RuntimeError("Failed to strictly load {}:\n{}".format(name, first_error))


def canonical_resnet_key(key: str) -> str:
    key = str(key)
    while key.startswith("module."):
        key = key[len("module."):]
    matches = []
    for root in _CANON_ROOTS:
        position = key.find(root)
        if position >= 0:
            matches.append((position, root))
    if not matches:
        return key
    position, _ = min(matches, key=lambda item: item[0])
    return key[position:]


def transplant_reference_teacher_state(source_teacher: nn.Module, ader_teacher: nn.Module) -> Dict:

    source = source_teacher.state_dict()
    target = ader_teacher.state_dict()
    by_canonical = {}
    for source_key, source_value in source.items():
        by_canonical.setdefault(canonical_resnet_key(source_key), []).append(
            (source_key, source_value)
        )

    mapped = {}
    ambiguous = {}
    unmatched = []
    new_state = dict(target)

    def is_executed(key: str) -> bool:
        return key.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3."))

    for target_key, target_value in target.items():
        canonical = canonical_resnet_key(target_key)
        if target_key in source and tuple(source[target_key].shape) == tuple(target_value.shape):
            candidates = [(target_key, source[target_key])]
        else:
            candidates = [
                (source_key, source_value)
                for source_key, source_value in by_canonical.get(canonical, [])
                if tuple(source_value.shape) == tuple(target_value.shape)
            ]
        if len(candidates) == 1:
            source_key, source_value = candidates[0]
            new_state[target_key] = source_value.detach().clone()
            mapped[target_key] = source_key
        elif len(candidates) > 1 and is_executed(canonical):
            ambiguous[target_key] = [item[0] for item in candidates]
        elif not candidates and is_executed(canonical):
            unmatched.append(target_key)

    if ambiguous:
        raise RuntimeError(
            "Ambiguous reference-to-ADer mapping for executed teacher tensors:\n"
            + json.dumps(ambiguous, indent=2)
        )
    if unmatched:
        raise RuntimeError(
            "Could not map all executed ADer teacher tensors:\n  "
            + "\n  ".join(unmatched[:50])
        )
    ader_teacher.load_state_dict(new_state, strict=True)
    return {
        "source_tensors": len(source),
        "target_tensors": len(target),
        "mapped_tensors": len(mapped),
        "unmatched_executed": unmatched,
    }


class SPRCRDModel(nn.Module):

    def __init__(
        self,
        model_t,
        model_s,
        *,
        backbone: str = "wide_resnet50_2",
        teacher_init: str = "reference_exact",
        teacher_pretrained: bool = True,
        teacher_checkpoint: str = "",
        model_checkpoint_path: str = "",
        embed_dims: Sequence[int] = (128, 160, 256),
        num_prototypes: Sequence[int] = (4, 4, 4),
        min_topk: Sequence[int] = (4, 2, 1),
        relation_weights: Sequence[float] = (0.15, 0.30, 0.45),
        temperature: float = 0.2,
        fusion_dim: int = 256,
        fusion_blocks: int = 2,
        norm: str = "bn",
        relation_warmup_epochs: int = 10,
        relation_ramp_epochs: int = 10,
        relation_momentum: float = 0.5,
        relation_smoothing: float = 1.0,
        quant_scale: float = 1e6,
        cf_posterior_temperature: float = 0.35,
        cf_relation_floor: float = 1e-4,
        cf_scale_weights: Sequence[float] = (0.15, 0.25, 0.60),
        cf_gain_margin: float = 0.08,
        cf_gain_temperature: float = 0.12,
        cf_max_intervention: float = 1.0,
        cf_gate_grad: bool = False,
        decoder_lr: float = 0.005,
        bottleneck_lr: float = 0.001,
    ):
        super().__init__()
        self.backbone = str(backbone)
        if self.backbone not in BACKBONE_SPECS:
            raise ValueError(
                "Unsupported SPRCRD backbone {!r}; supported={}".format(
                    self.backbone, tuple(BACKBONE_SPECS)
                )
            )
        feature_channels, decoder_in = BACKBONE_SPECS[self.backbone]
        
        
        
        model_t = copy.deepcopy(model_t)
        model_s = copy.deepcopy(model_s)
        model_t.name = "timm_{}".format(self.backbone)
        model_s.name = "de_{}".format(self.backbone)
        model_t.kwargs["features_only"] = True
        model_t.kwargs["out_indices"] = [1, 2, 3]
        self.net_t = get_model(model_t)
        self.net_s = get_model(model_s)
        self.bottleneck = StructuralPrototypeBottleneck(
            feature_channels=feature_channels,
            embed_dims=parse_triplet(embed_dims, int, "embed_dims"),
            num_prototypes=parse_triplet(num_prototypes, int, "num_prototypes"),
            min_topk=parse_triplet(min_topk, int, "min_topk"),
            relation_weights=parse_triplet(relation_weights, float, "relation_weights"),
            temperature=float(temperature),
            fusion_dim=int(fusion_dim),
            decoder_in_channels=int(decoder_in),
            fusion_blocks=int(fusion_blocks),
            relation_warmup_epochs=int(relation_warmup_epochs),
            relation_ramp_epochs=int(relation_ramp_epochs),
            relation_momentum=float(relation_momentum),
            relation_smoothing=float(relation_smoothing),
            norm=str(norm),
            quant_scale=float(quant_scale),
            cf_posterior_temperature=float(cf_posterior_temperature),
            cf_relation_floor=float(cf_relation_floor),
            cf_scale_weights=parse_triplet(cf_scale_weights, float, "cf_scale_weights"),
            cf_gain_margin=float(cf_gain_margin),
            cf_gain_temperature=float(cf_gain_temperature),
            cf_max_intervention=float(cf_max_intervention),
            cf_detach_gate=not bool(cf_gate_grad),
        )
        self.decoder_lr = float(decoder_lr)
        self.bottleneck_lr = float(bottleneck_lr)
        self._bottleneck_runner = None
        self.frozen_layers = ["net_t", "residual_calibrator"]
        self.residual_calibrator = None
        self.teacher_adapter_report = None
        self.checkpoint_metadata = {}

        checkpoint_payload = None
        if model_checkpoint_path:
            checkpoint_payload = _load_checkpoint_file(model_checkpoint_path)
            if not isinstance(checkpoint_payload, Mapping):
                raise TypeError("SPRC-RD checkpoint must be a dictionary")
            saved_backbone = checkpoint_payload.get("backbone")
            if saved_backbone and str(saved_backbone) != self.backbone:
                raise ValueError(
                    "checkpoint backbone={} but config backbone={}; "
                    "override model.kwargs.backbone to the checkpoint value".format(
                        saved_backbone, self.backbone
                    )
                )
            saved_args = checkpoint_payload.get("args", {})
            if isinstance(saved_args, Mapping):
                saved_prototypes = saved_args.get("num_prototypes")
                saved_topk = saved_args.get("min_topk")
                if saved_prototypes is not None and parse_triplet(
                    saved_prototypes, int, "checkpoint num_prototypes"
                ) != parse_triplet(num_prototypes, int, "num_prototypes"):
                    raise ValueError("checkpoint num_prototypes does not match config")
                if saved_topk is not None and parse_triplet(
                    saved_topk, int, "checkpoint min_topk"
                ) != parse_triplet(min_topk, int, "min_topk"):
                    raise ValueError("checkpoint min_topk does not match config")

        init_mode = str(teacher_init).lower()
        if checkpoint_payload is not None and isinstance(
            checkpoint_payload.get("ader_teacher_state"), Mapping
        ):
            init_mode = "ader_native"
        if init_mode not in ("reference_exact", "ader_native", "checkpoint_only"):
            raise ValueError("teacher_init must be reference_exact, ader_native, or checkpoint_only")

        if init_mode == "reference_exact":
            source_teacher = create_ader_teacher(
                backbone=self.backbone,
                pretrained=bool(teacher_pretrained) and not teacher_checkpoint,
                checkpoint_path=str(teacher_checkpoint),
                out_indices=(1, 2, 3),
            )
            if checkpoint_payload is not None and isinstance(checkpoint_payload.get("encoder"), Mapping):
                _strict_load(source_teacher, checkpoint_payload["encoder"], "reference encoder")
            self.teacher_adapter_report = transplant_reference_teacher_state(
                source_teacher, self.net_t
            )
            del source_teacher
        elif init_mode == "checkpoint_only":
            if checkpoint_payload is None or not isinstance(checkpoint_payload.get("encoder"), Mapping):
                raise ValueError("checkpoint_only requires model_checkpoint_path with encoder state")
            source_teacher = create_ader_teacher(
                backbone=self.backbone, pretrained=False, checkpoint_path="", out_indices=(1, 2, 3)
            )
            _strict_load(source_teacher, checkpoint_payload["encoder"], "reference encoder")
            self.teacher_adapter_report = transplant_reference_teacher_state(source_teacher, self.net_t)
            del source_teacher

        if checkpoint_payload is not None:
            if isinstance(checkpoint_payload.get("ader_teacher_state"), Mapping):
                _strict_load(
                    self.net_t,
                    checkpoint_payload["ader_teacher_state"],
                    "teacher encoder",
                )
            for key in ("bottleneck", "decoder"):
                if not isinstance(checkpoint_payload.get(key), Mapping):
                    raise KeyError("checkpoint is missing {!r}".format(key))
            _strict_load(self.bottleneck, checkpoint_payload["bottleneck"], "SPRC-RD bottleneck")
            _strict_load(self.net_s, checkpoint_payload["decoder"], "reverse decoder")
            if isinstance(checkpoint_payload.get("state_residual_stats"), Mapping):
                self.residual_calibrator = PrototypeStateResidualCalibrator.from_checkpoint(checkpoint_payload)
            self.checkpoint_metadata = {
                "path": str(model_checkpoint_path),
                "epoch": checkpoint_payload.get("epoch"),
                "method": checkpoint_payload.get("method"),
                "has_state_residual": self.residual_calibrator is not None,
            }

        self.freeze_layer(self.net_t)

    @staticmethod
    def freeze_layer(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        self.training = mode
        self.freeze_layer(self.net_t)
        self.bottleneck.train(mode)
        self.net_s.train(mode)
        if self.residual_calibrator is not None:
            self.freeze_layer(self.residual_calibrator)
        return self

    def get_optim_param_groups(self, _base_lr=None):
        return [
            {"params": self.net_s.parameters(), "lr": self.decoder_lr, "name": "decoder"},
            {"params": self.bottleneck.parameters(), "lr": self.bottleneck_lr, "name": "sprc_rd"},
        ]

    @property
    def encoder(self):
        return self.net_t

    @property
    def decoder(self):
        return self.net_s

    def set_bottleneck_runner(self, runner=None):
        self._bottleneck_runner = runner

    def forward_features(self, imgs: torch.Tensor = None, return_aux: bool = False,
                         teacher_override=None):
        if teacher_override is None:
            if imgs is None:
                raise ValueError("imgs is required when teacher_override is not supplied")
            feats_t = [feature.detach() for feature in self.net_t(imgs)]
        else:
            feats_t = [feature.detach() for feature in teacher_override]
        if self._bottleneck_runner is None:
            decoder_input, aux = self.bottleneck(feats_t, return_aux=return_aux)
        else:
            if bool(return_aux) != bool(self._bottleneck_runner.return_aux):
                raise ValueError("bottleneck runner return_aux mode mismatch")
            decoder_input, aux = self._bottleneck_runner(feats_t)
        feats_s = self.net_s(decoder_input)
        return feats_t, feats_s, aux

    def forward(self, imgs: torch.Tensor = None, return_aux: bool = False,
                teacher_override=None):
        feats_t, feats_s, aux = self.forward_features(
            imgs, return_aux=return_aux, teacher_override=teacher_override
        )
        return (feats_t, feats_s, aux) if return_aux else (feats_t, feats_s)

    def anomaly_map(self, imgs: torch.Tensor):
        if self.residual_calibrator is None:
            raise RuntimeError("No embedded prototype-state residual calibration statistics were loaded")
        feats_t, feats_s, aux = self.forward_features(imgs, return_aux=True)
        return self.residual_calibrator(feats_t, feats_s, aux, out_hw=tuple(imgs.shape[-2:]))

    def parameter_report(self):
        def count(module):
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(
                parameter.numel() for parameter in module.parameters() if parameter.requires_grad
            )
            return {"total": int(total), "trainable": int(trainable)}

        report = {
            "teacher": count(self.net_t),
            "bottleneck": count(self.bottleneck),
            "decoder": count(self.net_s),
        }
        report["full"] = {
            "total": sum(item["total"] for item in report.values()),
            "trainable": sum(item["trainable"] for item in report.values()),
        }
        return report


@MODEL.register_module
def sprc_rd(pretrained=False, **kwargs):
    del pretrained
    return SPRCRDModel(**kwargs)


__all__ = [
    "BACKBONE_SPECS",
    "SPRCRDModel",
    "canonical_resnet_key",
    "transplant_reference_teacher_state",
]
