from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import torch

ARCH_DEFAULTS = {
    "backbone": "wide_resnet50_2",
    "outer_impl": "ader",
    "teacher_checkpoint": "",
    "no_pretrained": False,
    "embed_dims": "128,160,256",
    "num_prototypes": "4,4,4",
    "min_topk": "4,2,1",
    "relation_weights": "0.15,0.30,0.45",
    "temperature": 0.2,
    "fusion_dim": 256,
    "fusion_blocks": 2,
    "norm": "bn",
    "relation_warmup_epochs": 10,
    "relation_ramp_epochs": 10,
    "relation_momentum": 0.5,
    "relation_smoothing": 1.0,
    "quant_scale": 1e6,
    "cf_posterior_temperature": 0.35,
    "cf_relation_floor": 1e-4,
    "cf_scale_weights": "0.15,0.25,0.60",
    "cf_gain_margin": 0.08,
    "cf_gain_temperature": 0.12,
    "cf_max_intervention": 1.0,
    "cf_gate_grad": False,
}


def load_checkpoint_file(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def architecture_from_checkpoint(checkpoint: Dict, outer_impl_override: str = "") -> SimpleNamespace:
    saved = checkpoint.get("args", {}) or {}
    values = dict(ARCH_DEFAULTS)
    values.update({key: saved[key] for key in values.keys() if key in saved and saved[key] is not None})
    if checkpoint.get("backbone"):
        values["backbone"] = checkpoint["backbone"]
    if checkpoint.get("outer_impl"):
        values["outer_impl"] = checkpoint["outer_impl"]
    if outer_impl_override:
        values["outer_impl"] = str(outer_impl_override)
    if "encoder" in checkpoint:
        values["no_pretrained"] = True
    return SimpleNamespace(**values)


def _strict_load(module, state, name: str) -> None:
    try:
        module.load_state_dict(state, strict=True)
        return
    except RuntimeError as first_error:
        if state and all(str(k).startswith("module.") for k in state):
            stripped = {str(k)[7:]: v for k, v in state.items()}
            try:
                module.load_state_dict(stripped, strict=True)
                return
            except RuntimeError:
                pass
        raise RuntimeError(f"Failed to strictly load {name}:\n{first_error}") from first_error


def load_model(path: str, device, freeze: bool = True, outer_impl_override: str = ""):
    checkpoint = load_checkpoint_file(path)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    missing = [key for key in ("bottleneck", "decoder") if key not in checkpoint]
    if missing:
        raise KeyError(f"incompatible checkpoint; missing keys: {missing}")
    config = architecture_from_checkpoint(checkpoint, outer_impl_override=outer_impl_override)

    from argparse import Namespace
    from model.sprc_rd import SPRCRDModel

    model_t = Namespace(
        name="timm_{}".format(config.backbone),
        kwargs=dict(
            pretrained=False,
            checkpoint_path="",
            strict=False,
            features_only=True,
            out_indices=[1, 2, 3],
        ),
    )
    model_s = Namespace(
        name="de_{}".format(config.backbone),
        kwargs=dict(pretrained=False, checkpoint_path="", strict=False),
    )
    architecture = {
        key: getattr(config, key)
        for key in ARCH_DEFAULTS
        if key not in ("backbone", "outer_impl", "no_pretrained", "teacher_checkpoint")
    }
    model = SPRCRDModel(
        model_t=model_t,
        model_s=model_s,
        backbone=config.backbone,
        teacher_init="reference_exact",
        teacher_pretrained=False,
        teacher_checkpoint=str(config.teacher_checkpoint),
        model_checkpoint_path=path,
        **architecture,
    ).to(device).eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, checkpoint, config
