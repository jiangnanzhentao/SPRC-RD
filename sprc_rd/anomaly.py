
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def rd_anomaly_map(teacher: Sequence[torch.Tensor], student: Sequence[torch.Tensor], out_hw: Tuple[int, int]):
    if len(teacher) != len(student):
        raise ValueError("teacher/student layer count mismatch")
    output = None
    layers = []
    for target, prediction in zip(teacher, student):
        amap = 1.0 - F.cosine_similarity(target, prediction, dim=1)
        amap = F.interpolate(amap[:, None], size=out_hw, mode="bilinear", align_corners=True)
        layers.append(amap)
        output = amap if output is None else output + amap
    return output, layers


def gaussian_blur(amap: torch.Tensor, sigma: float = 4.0):
    if sigma <= 0:
        return amap
    radius = max(1, int(round(3.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, device=amap.device, dtype=amap.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * float(sigma) ** 2))
    kernel = kernel / kernel.sum()
    channels = amap.shape[1]
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.pad(amap, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, ky, groups=channels)
