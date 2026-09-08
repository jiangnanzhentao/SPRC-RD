



















from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Sequence

import torch

from .encoder import (
    resnet18_encoder,
    resnet34_encoder,
    resnet50_encoder,
    wide_resnet50_2_encoder,
)





_LEGACY_ENCODERS = {
    "wide_resnet50_2": wide_resnet50_2_encoder,
    "resnet50": resnet50_encoder,
    "resnet34": resnet34_encoder,
    "resnet18": resnet18_encoder,
}




LEGACY_PRETRAINED_IDS = {
    "wide_resnet50_2": "torchvision:wide_resnet50_2-95faca4d.pth",
    "resnet50": "torchvision:resnet50-19c8e357.pth",
    "resnet34": "torchvision:resnet34-333f7ec4.pth",
    "resnet18": "torchvision:resnet18-5c106cde.pth",
}


def _unwrap_state_dict(obj):

    if not isinstance(obj, Mapping):
        return obj

    for key in ("encoder", "state_dict", "model", "model_state_dict"):
        value = obj.get(key)
        if isinstance(value, Mapping):
            obj = value
            break

    if isinstance(obj, Mapping) and obj and all(str(k).startswith("module.") for k in obj):
        obj = {str(k)[7:]: v for k, v in obj.items()}
    return obj


def _load_reference_checkpoint(model, checkpoint_path: str) -> None:





    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    state = _unwrap_state_dict(payload)
    if not isinstance(state, Mapping):
        raise TypeError(
            "teacher checkpoint must contain a PyTorch state_dict or an "
            "'encoder'/'state_dict' mapping"
        )
    model.load_state_dict(state, strict=True)


def create_ader_teacher(
    backbone: str,
    pretrained: bool = True,
    checkpoint_path: str = "",
    out_indices: Sequence[int] = (1, 2, 3),
):








    name = str(backbone)
    indices = tuple(int(x) for x in out_indices)
    if indices != (1, 2, 3):
        raise ValueError(
            "reference teacher exposes layer1/layer2/layer3 only; "
            f"expected out_indices=(1, 2, 3), got {indices}"
        )

    try:
        ctor = _LEGACY_ENCODERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported reference teacher backbone: {name!r}; "
            f"supported={tuple(_LEGACY_ENCODERS)}"
        ) from exc

    path = str(checkpoint_path or "").strip()

    
    
    
    use_builtin_pretrained = bool(pretrained) and not path
    model = ctor(pretrained=use_builtin_pretrained)

    if path:
        _load_reference_checkpoint(model, path)

    
    
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    
    model.teacher_impl = "reference_exact"
    model.teacher_pretrained_id = (
        os.path.abspath(path) if path else LEGACY_PRETRAINED_IDS.get(name, "")
    )
    return model


__all__ = ["create_ader_teacher", "LEGACY_PRETRAINED_IDS"]
