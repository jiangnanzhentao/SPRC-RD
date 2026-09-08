






from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6
DEPLOY_FORMAT_VERSION = 1

SCORING_DEFAULTS = {
    "state_known_mass_power": 0.25,
    "state_support_power": 0.5,
    "route_confidence_source": "product",
    "route_base_conf_power": 1.0,
    "route_entropy_power": 1.0,
    "route_reliability_boost": 1.5,
    "clnrm_score_mode": "hybrid_cdf",
    "clnrm_activation": "relu",
    "clnrm_tail_mix": 0.35,
    "clnrm_tail_clip": 4.0,
    "clnrm_cdf_mix": 0.70,
    "clnrm_cdf_clip": 8.0,
    "clnrm_cdf_eps": 1e-4,
    "clnrm_cdf_base_prob": 0.50,
    "clnrm_layer_weight": "dynamic_stable",
    "clnrm_layer_weights": "",
    "score_eps": 1e-6,
    "map_sigma": 4.0,
    "clnrm_map_sigma": -1.0,
}

REQUIRED_STATS = {
    "num_layers",
    "prototype_counts",
    "num_states",
    "state_triplets",
    "state_shrinkage",
    "state_q25",
    "state_q50",
    "state_q75",
    "state_q95",
    "state_q99",
    "state_cdf_values",
    "global_q25",
    "global_q50",
    "global_q75",
    "global_q95",
    "global_q99",
    "global_cdf_values",
    "global_inv_span",
    "global_layer_weights",
    "state_inv_span",
    "cdf_probs",
}

LONG_STATS = {"num_layers", "prototype_counts", "num_states", "state_triplets"}


def _to_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_payload(stats: Mapping[str, object]) -> None:
    missing = sorted(REQUIRED_STATS.difference(stats))
    if missing:
        raise KeyError("prototype-state residual calibration V3 stats missing: " + ", ".join(missing))
    if int(_to_numpy(stats["num_layers"]).reshape(-1)[0]) != 3:
        raise ValueError("prototype-state residual calibration V3 requires exactly three feature layers")
    if "uses_class_condition" in stats:
        if int(_to_numpy(stats["uses_class_condition"]).reshape(-1)[0]) != 0:
            raise ValueError("deployment runtime requires class-agnostic statistics")


def resolve_scoring_config(checkpoint: Mapping[str, object]) -> Dict[str, object]:
    values = dict(SCORING_DEFAULTS)
    deploy = checkpoint.get("state_residual_deploy", {})
    if isinstance(deploy, Mapping) and isinstance(deploy.get("args"), Mapping):
        source = deploy["args"]
    else:
        metadata = checkpoint.get("state_residual", {})
        source = metadata.get("args", {}) if isinstance(metadata, Mapping) else {}
    if isinstance(source, Mapping):
        for key in values:
            if key in source:
                values[key] = source[key]
    return values


def resolve_sigma(config: Mapping[str, object], override: float = -1.0) -> float:
    if float(override) >= 0:
        return float(override)
    calibrated = float(config.get("clnrm_map_sigma", -1.0))
    return float(config.get("map_sigma", 4.0)) if calibrated < 0 else calibrated


def _parse_weights(value) -> Sequence[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


class PrototypeStateResidualCalibrator(nn.Module):


    def __init__(self, stats: Mapping[str, object], config: Mapping[str, object], sigma: float):
        super().__init__()
        validate_payload(stats)
        merged = dict(SCORING_DEFAULTS)
        merged.update({key: config[key] for key in merged if key in config})
        self.config = SimpleNamespace(**merged)
        self.sigma = float(sigma)
        
        
        
        self.num_states_per_layer = tuple(
            int(value) for value in _to_numpy(stats["num_states"]).reshape(-1)[:3]
        )

        for name in sorted(REQUIRED_STATS):
            value = stats[name]
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
            else:
                tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(value)))
            tensor = tensor.long() if name in LONG_STATS else tensor.float()
            self.register_buffer(name, tensor, persistent=True)

        fixed_weights = self._make_fixed_layer_weights()
        self.register_buffer("fixed_layer_weights", fixed_weights, persistent=True)
        kernel = self._gaussian_kernel(self.sigma)
        self.register_buffer("gaussian_kernel", kernel, persistent=True)
        self.gaussian_radius = int((kernel.numel() - 1) // 2)

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, object], sigma: float = -1.0):
        stats = checkpoint.get("state_residual_stats")
        if not isinstance(stats, Mapping) or not stats:
            raise KeyError("checkpoint has no embedded state_residual_stats")
        config = resolve_scoring_config(checkpoint)
        return cls(stats, config, resolve_sigma(config, sigma))

    def _make_fixed_layer_weights(self) -> torch.Tensor:
        mode = str(self.config.clnrm_layer_weight).lower()
        if mode == "global_stable":
            weights = self.global_layer_weights.detach().float().clone().reshape(-1)[:3]
        elif mode == "custom":
            values = _parse_weights(self.config.clnrm_layer_weights)
            if len(values) != 3:
                raise ValueError("custom clnrm_layer_weights must contain three values")
            weights = torch.tensor(values, dtype=torch.float32)
        else:
            weights = torch.ones(3, dtype=torch.float32)
        weights = weights.clamp_min(0.0)
        return weights / weights.sum().clamp_min(EPS)

    @staticmethod
    def _gaussian_kernel(sigma: float) -> torch.Tensor:
        if float(sigma) <= 0:
            return torch.ones(1, dtype=torch.float32)
        radius = int(4.0 * float(sigma) + 0.5)
        positions = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (positions / float(sigma)).square())
        return kernel / kernel.sum()

    @staticmethod
    def _inclusive_reflect_pad(x: torch.Tensor, radius: int, dim: int) -> torch.Tensor:
        if radius <= 0:
            return x
        size = int(x.shape[dim])
        if radius > size:
            raise ValueError("Gaussian radius exceeds anomaly-map dimension")
        left_index = [slice(None)] * x.dim()
        right_index = [slice(None)] * x.dim()
        left_index[dim] = slice(0, radius)
        right_index[dim] = slice(size - radius, size)
        left = x[tuple(left_index)].flip(dim)
        right = x[tuple(right_index)].flip(dim)
        return torch.cat((left, x, right), dim=dim)

    def gaussian(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        if self.gaussian_radius == 0:
            return anomaly_map
        kernel = self.gaussian_kernel.to(dtype=anomaly_map.dtype)
        x = self._inclusive_reflect_pad(anomaly_map, self.gaussian_radius, -1)
        x = F.conv2d(x, kernel.view(1, 1, 1, -1))
        x = self._inclusive_reflect_pad(x, self.gaussian_radius, -2)
        return F.conv2d(x, kernel.view(1, 1, -1, 1))

    @staticmethod
    def _align_q(aux, name: str, target_hw: Tuple[int, int], dtype) -> torch.Tensor:
        q = aux[name]["cf_q"].to(dtype=dtype)
        if tuple(q.shape[1:3]) != tuple(target_hw):
            q = F.interpolate(
                q.permute(0, 3, 1, 2).contiguous(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()
        q = q.clamp_min(0.0)
        return q / q.sum(dim=-1, keepdim=True).clamp_min(EPS)

    def _route_confidence(self, aux, name: str, target_hw, dtype) -> torch.Tensor:
        native = aux[name]["cf_confidence"].to(dtype=dtype)
        verified = aux["prototype_state_confidence"].to(dtype=dtype)
        if tuple(native.shape[-2:]) != tuple(target_hw):
            native = F.interpolate(native, target_hw, mode="bilinear", align_corners=False)
        if tuple(verified.shape[-2:]) != tuple(target_hw):
            verified = F.interpolate(verified, target_hw, mode="bilinear", align_corners=False)
        native = native[:, 0].clamp(0.0, 1.0)
        verified = verified[:, 0].clamp(0.0, 1.0)
        source = str(self.config.route_confidence_source).lower()
        if source == "native":
            return native
        if source == "verified":
            return verified
        if source == "product":
            return torch.sqrt((native * verified).clamp_min(0.0)).clamp(0.0, 1.0)
        raise ValueError("unsupported route_confidence_source: " + source)

    def _route_reliability(self, q: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        qf = q.float().clamp_min(EPS)
        entropy = -(qf * qf.log()).sum(dim=-1)
        purity = (1.0 - entropy / math.log(float(max(int(q.shape[-1]), 2)))).clamp(0.0, 1.0)
        purity = purity.to(dtype=q.dtype).pow(float(self.config.route_entropy_power))
        base = confidence.clamp(0.0, 1.0).pow(float(self.config.route_base_conf_power))
        joint = torch.sqrt((base * purity).clamp_min(0.0)).clamp(0.0, 1.0)
        gamma = max(float(self.config.route_reliability_boost), EPS)
        return (1.0 - (1.0 - joint).clamp(0.0, 1.0).pow(gamma)).clamp(0.0, 1.0)

    def _state_router(self, aux, target_hw, dtype, layer: int):
        qs = []
        alphas = []
        for name in ("fine", "mid", "coarse"):
            q = self._align_q(aux, name, target_hw, dtype)
            confidence = self._route_confidence(aux, name, target_hw, dtype)
            qs.append(q)
            alphas.append(self._route_reliability(q, confidence))
        joint_alpha = (alphas[0] * alphas[1] * alphas[2]).clamp_min(0.0).pow(1.0 / 3.0)
        states = self.num_states_per_layer[layer]
        triplets = self.state_triplets[layer, :states]
        state_prob = (
            qs[0][..., triplets[:, 0]]
            * qs[1][..., triplets[:, 1]]
            * qs[2][..., triplets[:, 2]]
        ).clamp_min(0.0)
        known_mass = state_prob.sum(dim=-1).clamp(0.0, 1.0)
        weights = state_prob / known_mass.unsqueeze(-1).clamp_min(EPS)
        shrinkage = self.state_shrinkage[layer, :states].to(dtype=dtype)
        support = (weights * shrinkage.view(1, 1, 1, states)).sum(dim=-1).clamp(0.0, 1.0)
        mix = (
            joint_alpha
            * known_mass.pow(float(self.config.state_known_mass_power))
            * support.pow(float(self.config.state_support_power))
        ).clamp(0.0, 1.0)
        return weights, mix

    def _cdf_probability(self, residual, quantiles):
        batch, height, width = residual.shape
        q = torch.cummax(quantiles.to(dtype=residual.dtype), dim=-1).values
        probs = self.cdf_probs.to(dtype=residual.dtype)
        eps = max(float(self.config.clnrm_cdf_eps), EPS)
        probs = probs.clamp(eps, 1.0 - eps)
        curves, points = q.shape
        flat = residual.reshape(-1)
        count = flat.numel()
        expanded = flat.view(1, count).expand(curves, count).contiguous()
        indices = torch.searchsorted(q.contiguous(), expanded, right=False)
        index0 = (indices - 1).clamp(0, points - 1)
        index1 = indices.clamp(0, points - 1)
        q0 = q.gather(1, index0)
        q1 = q.gather(1, index1)
        prob_grid = probs.view(1, points).expand(curves, points)
        p0 = prob_grid.gather(1, index0)
        p1 = prob_grid.gather(1, index1)
        fraction = ((expanded - q0) / (q1 - q0).clamp_min(float(self.config.score_eps))).clamp(0.0, 1.0)
        cdf = p0 + fraction * (p1 - p0)

        below = indices <= 0
        first_q = q[:, :1].abs().clamp_min(float(self.config.score_eps)).expand(curves, count)
        lower = probs[0] * (expanded / first_q).clamp(0.0, 1.0)
        cdf = torch.where(below, lower, cdf)

        above = indices >= points
        if points >= 2:
            scale = (q[:, -1:] - q[:, -2:-1]).abs().clamp_min(float(self.config.score_eps))
        else:
            scale = q[:, -1:].abs().clamp_min(float(self.config.score_eps))
        extra = ((expanded - q[:, -1:]) / scale).clamp(0.0, 1.0)
        upper = probs[-1] + extra * ((1.0 - eps) - probs[-1])
        cdf = torch.where(above, upper, cdf).clamp(eps, 1.0 - eps)
        return cdf.transpose(0, 1).contiguous().view(batch, height, width, curves)

    def _tail_surprise(self, cdf):
        eps = max(float(self.config.clnrm_cdf_eps), EPS)
        base_prob = min(max(float(self.config.clnrm_cdf_base_prob), eps), 1.0 - eps)
        base_score = -math.log(max(1.0 - base_prob, eps))
        score = -torch.log1p(-cdf.clamp(eps, 1.0 - eps)) - base_score
        return F.relu(score).clamp(0.0, float(self.config.clnrm_cdf_clip))

    def core(self, teacher_features, student_features, aux, out_hw=None):
        if len(teacher_features) != 3 or len(student_features) != 3:
            raise ValueError("prototype-state residual calibration expects three teacher/student feature levels")
        if out_hw is None:
            out_hw = tuple(int(value) for value in teacher_features[0].shape[-2:])
        dtype = teacher_features[0].dtype
        score_maps = []
        reliability_maps = []

        for layer, (teacher, student) in enumerate(zip(teacher_features, student_features)):
            residual = 1.0 - F.cosine_similarity(teacher, student, dim=1)
            target_hw = tuple(int(value) for value in residual.shape[-2:])
            state_weight, mix = self._state_router(aux, target_hw, residual.dtype, layer)
            states = self.num_states_per_layer[layer]
            residual4 = residual.unsqueeze(-1)

            q25 = self.state_q25[layer, :states].to(dtype=dtype)
            q50 = self.state_q50[layer, :states].to(dtype=dtype)
            q75 = self.state_q75[layer, :states].to(dtype=dtype)
            q95 = self.state_q95[layer, :states].to(dtype=dtype)
            q99 = self.state_q99[layer, :states].to(dtype=dtype)
            state_iqr = (residual4 - q50.view(1, 1, 1, states)) / (
                q75 - q25
            ).clamp_min(float(self.config.score_eps)).view(1, 1, 1, states)
            activation = str(self.config.clnrm_activation).lower()
            if activation == "relu":
                state_iqr = F.relu(state_iqr)
            elif activation == "softplus":
                state_iqr = F.relu(F.softplus(state_iqr) - math.log(2.0))
            elif activation != "identity":
                raise ValueError("unsupported clnrm_activation: " + activation)
            state_tail = F.relu(
                (residual4 - q95.view(1, 1, 1, states))
                / (q99 - q50).clamp_min(float(self.config.score_eps)).view(1, 1, 1, states)
            ).clamp(0.0, float(self.config.clnrm_tail_clip))
            state_quantile = (
                (1.0 - float(self.config.clnrm_tail_mix)) * state_iqr
                + float(self.config.clnrm_tail_mix) * state_tail
            )
            iqr_state = (state_weight * state_iqr).sum(dim=-1)
            tail_state = (state_weight * state_tail).sum(dim=-1)
            quantile_state = (state_weight * state_quantile).sum(dim=-1)

            g25 = self.global_q25[layer].to(dtype=dtype)
            g50 = self.global_q50[layer].to(dtype=dtype)
            g75 = self.global_q75[layer].to(dtype=dtype)
            g95 = self.global_q95[layer].to(dtype=dtype)
            g99 = self.global_q99[layer].to(dtype=dtype)
            iqr_global = (residual - g50) / (g75 - g25).clamp_min(float(self.config.score_eps))
            if activation == "relu":
                iqr_global = F.relu(iqr_global)
            elif activation == "softplus":
                iqr_global = F.relu(F.softplus(iqr_global) - math.log(2.0))
            tail_global = F.relu(
                (residual - g95) / (g99 - g50).clamp_min(float(self.config.score_eps))
            ).clamp(0.0, float(self.config.clnrm_tail_clip))
            quantile_global = (
                (1.0 - float(self.config.clnrm_tail_mix)) * iqr_global
                + float(self.config.clnrm_tail_mix) * tail_global
            )

            iqr = mix * iqr_state + (1.0 - mix) * iqr_global
            tail = mix * tail_state + (1.0 - mix) * tail_global
            quantile = mix * quantile_state + (1.0 - mix) * quantile_global
            mode = str(self.config.clnrm_score_mode).lower()
            if mode == "ziqr":
                score = iqr
            elif mode == "tail":
                score = tail
            elif mode == "quantile":
                score = quantile
            elif mode in ("cdf", "hybrid_cdf"):
                state_curves = self.state_cdf_values[layer, :states].to(dtype=dtype)
                state_cdf = self._cdf_probability(residual, state_curves)
                state_cdf = (state_weight * state_cdf).sum(dim=-1)
                global_curve = self.global_cdf_values[layer : layer + 1].to(dtype=dtype)
                global_cdf = self._cdf_probability(residual, global_curve)[..., 0]
                cdf = mix * state_cdf + (1.0 - mix) * global_cdf
                cdf_score = self._tail_surprise(cdf)
                score = cdf_score if mode == "cdf" else (
                    (1.0 - float(self.config.clnrm_cdf_mix)) * quantile
                    + float(self.config.clnrm_cdf_mix) * cdf_score
                )
            else:
                raise ValueError("unsupported clnrm_score_mode: " + mode)

            score_maps.append(
                F.interpolate(score.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False)
            )
            inverse_state = self.state_inv_span[layer, :states].to(dtype=dtype)
            reliability_state = (
                state_weight * inverse_state.view(1, 1, 1, states)
            ).sum(dim=-1)
            reliability_global = self.global_inv_span[layer].to(dtype=dtype)
            reliability = mix * reliability_state + (1.0 - mix) * reliability_global
            reliability_maps.append(
                F.interpolate(reliability.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False)
            )

        stack = torch.stack(score_maps, dim=0)
        if str(self.config.clnrm_layer_weight).lower() == "dynamic_stable":
            reliability = torch.stack(reliability_maps, dim=0)
            weights = reliability / reliability.sum(dim=0, keepdim=True).clamp_min(EPS)
            return (weights * stack).sum(dim=0)
        weights = self.fixed_layer_weights.to(dtype=dtype).view(3, 1, 1, 1, 1)
        return (weights * stack).sum(dim=0)

    def forward(self, teacher_features, student_features, aux, out_hw=None):
        return self.gaussian(self.core(teacher_features, student_features, aux, out_hw=out_hw))


__all__ = [
    "DEPLOY_FORMAT_VERSION",
    "SCORING_DEFAULTS",
    "PrototypeStateResidualCalibrator",
    "resolve_scoring_config",
    "resolve_sigma",
    "validate_payload",
]