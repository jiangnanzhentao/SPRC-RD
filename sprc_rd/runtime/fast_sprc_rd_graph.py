







from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fast_sprc_rd_v3 as v3
from . import fast_sprc_rd_v4 as v4
from .cuda_graph import CUDAGraphBottleneck


class FastStructuralPrototypeBottleneckWithCalibration(v4.FastStructuralPrototypeBottleneckV4):


    def forward(self, features: Sequence[torch.Tensor], return_aux: bool = False):
        if not return_aux:
            return super().forward(features, return_aux=False)
        if len(features) != 3:
            raise ValueError("expected exactly three teacher feature tensors")

        fine_obs, g1, c1, q1 = self.scales[0](
            features[0], self.cf_posterior_temperature, self.cf_relation_floor, True
        )
        mid_obs, g2, c2, q2 = self.scales[1](
            features[1], self.cf_posterior_temperature, self.cf_relation_floor, True
        )
        coarse_obs, g3, c3, q3 = self.scales[2](
            features[2], self.cf_posterior_temperature, self.cf_relation_floor, True
        )

        fine_scale, mid_scale, coarse_scale = self.scales
        fine = fine_obs if fine_scale.output_folded else fine_scale.output_norm(fine_obs)
        mid = mid_obs if mid_scale.output_folded else mid_scale.output_norm(mid_obs)

        coarse_basis = (
            coarse_scale.proto_out if coarse_scale.output_folded else coarse_scale.proto_n
        )
        if coarse_basis.dtype != q3.dtype:
            coarse_basis = coarse_basis.to(dtype=q3.dtype)
        coarse_cf = torch.matmul(q3, coarse_basis).permute(0, 3, 1, 2).contiguous()

        coarse_hw = coarse_obs.shape[-2:]
        if g1.shape[-2:] == (coarse_hw[0] * 4, coarse_hw[1] * 4):
            g1a = F.avg_pool2d(g1, 4, 4)
            c1a = F.avg_pool2d(c1, 4, 4)
        else:
            g1a = F.adaptive_avg_pool2d(g1, coarse_hw)
            c1a = F.adaptive_avg_pool2d(c1, coarse_hw)

        if g2.shape[-2:] == (coarse_hw[0] * 2, coarse_hw[1] * 2):
            g2a = F.avg_pool2d(g2, 2, 2)
            c2a = F.avg_pool2d(c2, 2, 2)
        else:
            g2a = F.adaptive_avg_pool2d(g2, coarse_hw)
            c2a = F.adaptive_avg_pool2d(c2, coarse_hw)

        weights = self.cf_scale_weights_fast.to(device=g3.device, dtype=g3.dtype)
        hier_gain = weights[0] * g1a + weights[1] * g2a + weights[2] * g3
        hier_conf = weights[0] * c1a + weights[1] * c2a + weights[2] * c3
        verified_conf = (c3 * hier_conf).sqrt().clamp(0.0, 1.0)

        excess = F.relu(hier_gain - self.cf_gain_margin)
        gain_activation = 1.0 - torch.exp(-excess / self.cf_gain_temperature)
        alpha = float(self.scales[2].relation_alpha)
        gate = (
            alpha * self.cf_max_intervention * verified_conf * gain_activation
        ).clamp(0.0, self.cf_max_intervention)
        if self.cf_detach_gate:
            gate = gate.detach()

        coarse = coarse_obs + gate * (coarse_cf - coarse_obs)
        if not coarse_scale.output_folded:
            coarse = coarse_scale.output_norm(coarse)

        fine_aligned = self.fine_down2(self.fine_down1(fine))
        mid_aligned = self.mid_down(mid)
        fused = self.fuse(torch.cat([fine_aligned, mid_aligned, coarse], dim=1))
        fused = self.mixers(fused)
        fused = self.to_8(fused)
        decoder_input = self.decoder_proj(fused).contiguous()

        aux = {
            "fine": {"cf_q": q1, "cf_confidence": c1},
            "mid": {"cf_q": q2, "cf_confidence": c2},
            "coarse": {"cf_q": q3, "cf_confidence": c3},
            "prototype_state_confidence": verified_conf,
        }
        return decoder_input, aux


def _audit_tensor(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    stats = v3._tensor_error(reference, candidate)
    stats["pass"] = bool(v3._passes(stats, 1e-4, 1e-5, 0.999999))
    return stats


def _required_aux(aux):
    return {
        "fine.cf_q": aux["fine"]["cf_q"],
        "mid.cf_q": aux["mid"]["cf_q"],
        "coarse.cf_q": aux["coarse"]["cf_q"],
        "fine.cf_confidence": aux["fine"]["cf_confidence"],
        "mid.cf_confidence": aux["mid"]["cf_confidence"],
        "coarse.cf_confidence": aux["coarse"]["cf_confidence"],
        "prototype_state_confidence": aux["prototype_state_confidence"],
    }


def build_v4_drop_i32_cudagraph(
    original_bottleneck: nn.Module,
    example_features: Sequence[torch.Tensor],
    *,
    return_aux: bool = True,
    audit: bool = True,
    fuse_bn: bool = True,
    fold_output_bn: bool = True,
    capture_warmup: int = 5,
):









    if len(example_features) != 3:
        raise ValueError("example_features must contain three tensors")
    if any(x.device.type != "cuda" for x in example_features):
        raise RuntimeError("v4_drop_i32 CUDA Graph requires CUDA tensors")

    original_bottleneck.eval()
    feature_hws = [tuple(int(v) for v in x.shape[-2:]) for x in example_features]
    fast_cls = FastStructuralPrototypeBottleneckWithCalibration if return_aux else v4.FastStructuralPrototypeBottleneckV4
    fast = fast_cls(
        original_bottleneck,
        feature_hws,
        fuse_bn=fuse_bn,
        fold_output_bn=fold_output_bn,
        sparse_backend="drop_i32",
    ).to(example_features[0].device).eval()

    report = {}
    if audit:
        with torch.inference_mode():
            ref_out, ref_aux = original_bottleneck(example_features, return_aux=return_aux)
            fast_out, fast_aux = fast(example_features, return_aux=return_aux)
            torch.cuda.synchronize(example_features[0].device)

        report["v4_output"] = _audit_tensor(ref_out, fast_out)
        if not report["v4_output"]["pass"]:
            raise RuntimeError("v4_drop_i32 output audit failed: {}".format(report["v4_output"]))

        if return_aux:
            ref_required = _required_aux(ref_aux)
            got_required = _required_aux(fast_aux)
            report["v4_aux"] = {
                key: _audit_tensor(ref_required[key], got_required[key])
                for key in ref_required
            }
            failed = {k: v for k, v in report["v4_aux"].items() if not v["pass"]}
            if failed:
                raise RuntimeError("v4_drop_i32 aux audit failed: {}".format(failed))

    runner = CUDAGraphBottleneck(
        fast,
        example_features,
        return_aux=return_aux,
        warmup=capture_warmup,
    )
    runner.fast_v4_module = fast
    runner.backend = "v4_drop_i32_cudagraph"

    if audit:
        with torch.inference_mode():
            graph_out, graph_aux = runner(example_features)
            torch.cuda.synchronize(example_features[0].device)

        report["graph_output"] = _audit_tensor(ref_out, graph_out)
        if not report["graph_output"]["pass"]:
            raise RuntimeError(
                "v4_drop_i32 CUDA Graph output audit failed: {}".format(
                    report["graph_output"]
                )
            )

        if return_aux:
            graph_required = _required_aux(graph_aux)
            report["graph_aux"] = {
                key: _audit_tensor(ref_required[key], graph_required[key])
                for key in ref_required
            }
            failed = {k: v for k, v in report["graph_aux"].items() if not v["pass"]}
            if failed:
                raise RuntimeError("v4_drop_i32 CUDA Graph aux audit failed: {}".format(failed))

    return runner, report


__all__ = [
    "FastStructuralPrototypeBottleneckWithCalibration",
    "build_v4_drop_i32_cudagraph",
]
