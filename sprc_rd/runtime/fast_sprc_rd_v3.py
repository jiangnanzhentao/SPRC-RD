
















































































from __future__ import annotations

import copy
import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.fusion import fuse_conv_bn_eval
except Exception:
    fuse_conv_bn_eval = None


_EPS = 1e-6
_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_error(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    aa = a.detach().float()
    bb = b.detach().float()
    if aa.shape != bb.shape:
        return {
            "shape_match": False,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "cosine": -1.0,
        }
    diff = (aa - bb).abs()
    flat_a = aa.reshape(aa.shape[0], -1)
    flat_b = bb.reshape(bb.shape[0], -1)
    cos = F.cosine_similarity(flat_a, flat_b, dim=1).mean()
    return {
        "shape_match": True,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cosine": float(cos.item()),
    }


def _passes(
    stats: Dict[str, float],
    max_abs: float,
    mean_abs: float,
    min_cosine: float,
) -> bool:
    return bool(
        stats["shape_match"]
        and stats["max_abs"] <= max_abs
        and stats["mean_abs"] <= mean_abs
        and stats["cosine"] >= min_cosine
    )


def _deterministic_order(prob: torch.Tensor, quant_scale: float) -> torch.Tensor:
    p = prob.detach().float()
    k = p.shape[-1]
    q = torch.round(p * float(quant_scale)).to(torch.int64)
    ids = torch.arange(k, device=p.device, dtype=torch.int64)
    key = q * (k + 1) + (k - ids)
    return torch.argsort(key, dim=-1, descending=True)


def _fuse_convnormact(module: nn.Module) -> nn.Module:

    m = copy.deepcopy(module).eval()
    if fuse_conv_bn_eval is None:
        return m
    if not isinstance(m, nn.Sequential) or len(m) < 2:
        return m
    if isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.BatchNorm2d):
        fused = fuse_conv_bn_eval(m[0], m[1])
        layers: List[nn.Module] = [fused]
        for i in range(2, len(m)):
            layers.append(m[i])
        return nn.Sequential(*layers)
    return m


def _fuse_depthwise_downsample(module: nn.Module) -> nn.Module:
    m = copy.deepcopy(module).eval()
    if hasattr(m, "dw"):
        m.dw = _fuse_convnormact(m.dw)
    if hasattr(m, "pw"):
        m.pw = _fuse_convnormact(m.pw)
    return m


def _fuse_mixer(module: nn.Module) -> nn.Module:
    m = copy.deepcopy(module).eval()
    if hasattr(m, "dw"):
        m.dw = _fuse_convnormact(m.dw)
    if (
        fuse_conv_bn_eval is not None
        and hasattr(m, "pw2")
        and hasattr(m, "out_norm")
        and isinstance(m.pw2, nn.Conv2d)
        and isinstance(m.out_norm, nn.BatchNorm2d)
    ):
        m.pw2 = fuse_conv_bn_eval(m.pw2, m.out_norm)
        m.out_norm = nn.Identity()
    return m


def _fuse_seq_conv_bn(module: nn.Module) -> nn.Module:
    m = copy.deepcopy(module).eval()
    if fuse_conv_bn_eval is None:
        return m
    if isinstance(m, nn.Sequential) and len(m) >= 2:
        if isinstance(m[0], nn.Conv2d) and isinstance(m[1], nn.BatchNorm2d):
            fused = fuse_conv_bn_eval(m[0], m[1])
            layers = [fused] + [m[i] for i in range(2, len(m))]
            return nn.Sequential(*layers)
    return m


def _make_post_modules(src: nn.Module, fuse_bn: bool) -> Dict[str, nn.Module]:
    if not fuse_bn:
        return {
            "fine_down1": copy.deepcopy(src.fine_down1).eval(),
            "fine_down2": copy.deepcopy(src.fine_down2).eval(),
            "mid_down": copy.deepcopy(src.mid_down).eval(),
            "fuse": copy.deepcopy(src.fuse).eval(),
            "mixers": copy.deepcopy(src.mixers).eval(),
            "to_8": copy.deepcopy(src.to_8).eval(),
            "decoder_proj": copy.deepcopy(src.decoder_proj).eval(),
        }

    mixers = copy.deepcopy(src.mixers).eval()
    if isinstance(mixers, nn.Sequential):
        mixers = nn.Sequential(*[_fuse_mixer(m) for m in mixers])

    return {
        "fine_down1": _fuse_depthwise_downsample(src.fine_down1),
        "fine_down2": _fuse_depthwise_downsample(src.fine_down2),
        "mid_down": _fuse_depthwise_downsample(src.mid_down),
        "fuse": _fuse_convnormact(src.fuse),
        "mixers": mixers,
        "to_8": _fuse_depthwise_downsample(src.to_8),
        "decoder_proj": _fuse_seq_conv_bn(src.decoder_proj),
    }


def _bn_folded_prototypes(
    proto_n: torch.Tensor,
    output_norm: nn.Module,
) -> Optional[torch.Tensor]:









    if not isinstance(output_norm, nn.BatchNorm2d):
        return None
    bn = copy.deepcopy(output_norm).eval()
    if bn.running_mean is None or bn.running_var is None:
        return None

    dtype = proto_n.dtype
    device = proto_n.device
    running_mean = bn.running_mean.detach().to(device=device, dtype=torch.float32)
    running_var = bn.running_var.detach().to(device=device, dtype=torch.float32)
    if bn.affine:
        gamma = bn.weight.detach().to(device=device, dtype=torch.float32)
        beta = bn.bias.detach().to(device=device, dtype=torch.float32)
    else:
        gamma = torch.ones_like(running_mean)
        beta = torch.zeros_like(running_mean)

    scale = gamma / torch.sqrt(running_var + float(bn.eps))
    bias = beta - running_mean * scale
    folded = proto_n.float() * scale.view(1, -1) + bias.view(1, -1)
    return folded.to(dtype=dtype)


def _geometry(
    h: int,
    w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:






    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.long),
        torch.arange(w, device=device, dtype=torch.long),
        indexing="ij",
    )
    neigh_idx = []
    prev_idx = []
    valid_out = []
    valid_in = []

    for dy, dx in _DIRECTIONS:
        ny = yy + int(dy)
        nx = xx + int(dx)
        vo = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ni = ny.clamp(0, h - 1)
        nj = nx.clamp(0, w - 1)
        out_idx = (ni * w + nj).reshape(-1)
        out_idx = torch.where(vo.reshape(-1), out_idx, torch.zeros_like(out_idx))

        py = yy - int(dy)
        px = xx - int(dx)
        vi = (py >= 0) & (py < h) & (px >= 0) & (px < w)
        pi = py.clamp(0, h - 1)
        pj = px.clamp(0, w - 1)
        in_idx = (pi * w + pj).reshape(-1)
        in_idx = torch.where(vi.reshape(-1), in_idx, torch.zeros_like(in_idx))

        neigh_idx.append(out_idx)
        prev_idx.append(in_idx)
        valid_out.append(vo.reshape(-1))
        valid_in.append(vi.reshape(-1))

    neigh_idx_t = torch.stack(neigh_idx, dim=0)
    prev_idx_t = torch.stack(prev_idx, dim=0)
    valid_out_t = torch.stack(valid_out, dim=0)
    valid_in_t = torch.stack(valid_in, dim=0)
    valid_obs_count = valid_out_t.sum(dim=0).clamp_min(1).to(torch.float32)
    valid_cf_count = (valid_out_t.sum(dim=0) + valid_in_t.sum(dim=0)).clamp_min(1).to(torch.float32)
    return (
        neigh_idx_t,
        prev_idx_t,
        valid_out_t,
        valid_in_t,
        valid_obs_count,
        valid_cf_count,
    )


class _FastScaleBase(nn.Module):
    def __init__(
        self,
        src: nn.Module,
        hw: Tuple[int, int],
        *,
        fuse_bn: bool,
        fold_output_bn: bool,
    ) -> None:
        super().__init__()
        src.eval()

        self.embed_dim = int(src.embed_dim)
        self.num_prototypes = int(src.num_prototypes)
        self.temperature = float(src.temperature)
        self.min_topk = int(src.min_topk)
        self.relation_weight = float(src.relation_weight)
        self.relation_log_floor = float(src.relation_log_floor)
        self.quant_scale = float(src.quant_scale)
        self.relation_alpha = float(src.relation_alpha)
        self.h = int(hw[0])
        self.w = int(hw[1])
        self.n = self.h * self.w
        self.log_floor = float(math.exp(self.relation_log_floor))
        self.log_num_prototypes = float(math.log(float(self.num_prototypes)))

        self.adapter = _fuse_convnormact(src.adapter) if fuse_bn else copy.deepcopy(src.adapter).eval()

        proto_n = F.normalize(src.prototypes.detach().float(), dim=-1, eps=_EPS)
        relation_prob = src.relation_prob.detach().float()
        row_max = relation_prob.max(dim=-1).values.clamp_min(_EPS)
        trans_norm = (relation_prob / row_max.unsqueeze(-1)).clamp(0.0, 1.0)
        gates = src.relation_gate().detach().float()

        folded = None
        if fold_output_bn:
            folded = _bn_folded_prototypes(proto_n, src.output_norm)

        if folded is not None:
            self.output_folded = True
            self.output_norm = nn.Identity()
            self.register_buffer("proto_out", folded, persistent=False)
        else:
            self.output_folded = False
            self.output_norm = copy.deepcopy(src.output_norm).eval()
            self.register_buffer("proto_out", proto_n, persistent=False)

        self.register_buffer("proto_n", proto_n, persistent=False)
        self.register_buffer("relation_prob_fast", relation_prob, persistent=False)
        self.register_buffer("row_max_fast", row_max, persistent=False)
        self.register_buffer("trans_norm_fast", trans_norm, persistent=False)
        self.register_buffer("gates_fast", gates, persistent=False)

        geom = _geometry(self.h, self.w, relation_prob.device)
        self.register_buffer("neigh_idx", geom[0], persistent=False)
        self.register_buffer("prev_idx", geom[1], persistent=False)
        self.register_buffer("valid_out", geom[2], persistent=False)
        self.register_buffer("valid_in", geom[3], persistent=False)
        self.register_buffer("valid_obs_count", geom[4], persistent=False)
        self.register_buffer("valid_cf_count", geom[5], persistent=False)

    def _embed_and_appearance(self, x: torch.Tensor):
        z = self.adapter(x)
        b, d, h, w = z.shape
        if h != self.h or w != self.w:
            raise ValueError(
                "Fast scale was prepared for {}x{}, got {}x{}".format(
                    self.h, self.w, h, w
                )
            )
        tokens = z.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_norm_forward(tokens)

        
        
        
        embedding = F.normalize(tokens.float(), dim=-1, eps=_EPS).to(dtype=tokens.dtype)
        embedding = embedding.transpose(1, 2).contiguous().view(b, d, h, w)
        tokens_n = F.normalize(embedding.float(), dim=1, eps=_EPS)
        tokens_n = tokens_n.flatten(2).transpose(1, 2).contiguous().to(dtype=embedding.dtype)

        proto_n = self.proto_n
        if proto_n.dtype != tokens_n.dtype:
            proto_n = proto_n.to(dtype=tokens_n.dtype)
        appearance_logits = torch.matmul(tokens_n, proto_n.t()) / self.temperature
        appearance_prob = F.softmax(appearance_logits, dim=-1)
        return z, appearance_logits, appearance_prob

    def token_norm_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_norm(tokens)

    def _sparse_projection(
        self,
        prob: torch.Tensor,
        suspicion: torch.Tensor,
    ) -> torch.Tensor:
        k = self.num_prototypes
        if self.relation_alpha <= 0.0 or self.min_topk >= k:
            return prob

        k_float = k - self.relation_alpha * suspicion * float(k - self.min_topk)
        topk = torch.round(k_float).to(torch.long).clamp_(self.min_topk, k)

        
        order = _deterministic_order(prob, self.quant_scale)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(k, device=prob.device, dtype=order.dtype)
        rank_values = rank_values.view(*([1] * (order.dim() - 1)), k).expand_as(order)
        ranks.scatter_(-1, order, rank_values)
        mask = ranks < topk.unsqueeze(-1)

        sparse = prob * mask.to(dtype=prob.dtype)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(_EPS)

    def _finish_projection(
        self,
        appearance_logits: torch.Tensor,
        appearance_prob: torch.Tensor,
        rel_flat: torch.Tensor,
        structure_strength: torch.Tensor,
        relation_consistency: torch.Tensor,
    ) -> torch.Tensor:
        b = appearance_prob.shape[0]
        final_logits = appearance_logits + (
            self.relation_alpha
            * self.relation_weight
            * rel_flat
        )
        final_prob = F.softmax(final_logits, dim=-1)
        suspicion = (
            structure_strength * (1.0 - relation_consistency)
        )
        sparse_prob = self._sparse_projection(final_prob, suspicion)

        proto_out = self.proto_out
        if proto_out.dtype != sparse_prob.dtype:
            proto_out = proto_out.to(dtype=sparse_prob.dtype)
        normal_tokens = torch.matmul(sparse_prob, proto_out)
        out = normal_tokens.transpose(1, 2).contiguous().view(
            b, self.embed_dim, self.h, self.w
        )
        
        
        
        return out


class FastScaleLoop(_FastScaleBase):

    def __init__(self, src, hw, *, fuse_bn: bool, fold_output_bn: bool):
        super().__init__(
            src, hw,
            fuse_bn=fuse_bn,
            fold_output_bn=fold_output_bn,
        )
        self.token_norm = copy.deepcopy(src.token_norm).eval()

    @staticmethod
    def _slices(h, w, dy, dx):
        if dy >= 0:
            cy0, cy1, ny0, ny1 = 0, h-dy, dy, h
        else:
            cy0, cy1, ny0, ny1 = -dy, h, 0, h+dy
        if dx >= 0:
            cx0, cx1, nx0, nx1 = 0, w-dx, dx, w
        else:
            cx0, cx1, nx0, nx1 = -dx, w, 0, w+dx
        return (slice(cy0, cy1), slice(cx0, cx1)), (slice(ny0, ny1), slice(nx0, nx1))

    def forward(self, x, posterior_temperature: float, relation_floor: float, need_q: bool):
        _, appearance_logits, appearance_prob = self._embed_and_appearance(x)
        b = appearance_prob.shape[0]
        h, w, k = self.h, self.w, self.num_prototypes
        app = appearance_prob.view(b, h, w, k)

        rel = app.new_zeros((b, h, w, k))
        strength_num = app.new_zeros((b, h, w))
        consistency_num = app.new_zeros((b, h, w))
        reliability_den = app.new_zeros((b, h, w))
        valid_count = app.new_zeros((1, h, w))

        energy_num = app.new_zeros((b, h, w, k))
        weight_num = app.new_zeros((b, h, w, k))
        cf_valid_count = app.new_zeros((1, h, w, 1))

        gates_all = self.gates_fast.to(dtype=app.dtype)
        rel_prob_all = self.relation_prob_fast.to(dtype=app.dtype)
        row_max_all = self.row_max_fast.to(dtype=app.dtype)
        trans_norm_all = self.trans_norm_fast.to(dtype=app.dtype)
        floor = max(float(relation_floor), _EPS)

        for di, (dy, dx) in enumerate(_DIRECTIONS):
            (cy, cx), (ny, nx) = self._slices(h, w, dy, dx)
            center_app = app[:, cy, cx, :]
            neigh_app = app[:, ny, nx, :]
            gate = gates_all[di]
            rel_prob = rel_prob_all[di]
            row_max = row_max_all[di]
            trans_norm = trans_norm_all[di]

            compat_raw = torch.matmul(neigh_app, rel_prob.t())
            compat_obs = (
                compat_raw / row_max.view(1, 1, 1, k)
            ).clamp(0.0, 1.0)

            log_compat = compat_obs.clamp_min(self.log_floor).log()
            rel[:, cy, cx, :] += log_compat * gate.view(1, 1, 1, k)

            expected_gate = (
                center_app * gate.view(1, 1, 1, k)
            ).sum(dim=-1)
            expected_gate_compat = (
                center_app * gate.view(1, 1, 1, k) * compat_obs
            ).sum(dim=-1)
            strength_num[:, cy, cx] += expected_gate
            consistency_num[:, cy, cx] += expected_gate_compat
            reliability_den[:, cy, cx] += expected_gate
            valid_count[:, cy, cx] += 1.0

            
            compat_fwd = compat_obs.clamp(floor, 1.0)
            rel_fwd = gate.view(1, 1, 1, k).expand_as(compat_fwd)
            energy_num[:, cy, cx, :] += rel_fwd * (-compat_fwd.log())
            weight_num[:, cy, cx, :] += rel_fwd
            cf_valid_count[:, cy, cx, :] += 1.0

            weighted_center = center_app * gate.view(1, 1, 1, k)
            reverse_rel = weighted_center.sum(dim=-1, keepdim=True)
            reverse_num = torch.matmul(weighted_center, trans_norm)
            compat_rev = (
                reverse_num / reverse_rel.clamp_min(_EPS)
            ).clamp(floor, 1.0)
            rel_rev = reverse_rel.expand_as(compat_rev)
            energy_num[:, ny, nx, :] += rel_rev * (-compat_rev.log())
            weight_num[:, ny, nx, :] += rel_rev
            cf_valid_count[:, ny, nx, :] += 1.0

        valid = valid_count.clamp_min(1.0)
        rel = rel / valid.unsqueeze(-1)
        structure_strength = (strength_num / valid).clamp(0.0, 1.0)
        relation_consistency = torch.where(
            reliability_den > _EPS,
            consistency_num / reliability_den.clamp_min(_EPS),
            torch.ones_like(reliability_den),
        ).clamp(0.0, 1.0)

        observed = self._finish_projection(
            appearance_logits,
            appearance_prob,
            rel.view(b, h*w, k),
            structure_strength.view(b, h*w),
            relation_consistency.view(b, h*w),
        )

        candidate_energy = torch.where(
            weight_num > _EPS,
            energy_num / weight_num.clamp_min(_EPS),
            torch.zeros_like(energy_num),
        )
        q_cf = F.softmax(
            -candidate_energy.float() / float(posterior_temperature),
            dim=-1,
        ).to(dtype=app.dtype)

        e_obs = (app * candidate_energy).sum(dim=-1, keepdim=True)
        e_cf = (q_cf * candidate_energy).sum(dim=-1, keepdim=True)
        gain = e_obs - e_cf

        qf = q_cf.float().clamp_min(_EPS)
        entropy = -(qf * qf.log()).sum(dim=-1, keepdim=True)
        certainty = (
            1.0 - entropy / self.log_num_prototypes
        ).clamp(0.0, 1.0).to(dtype=app.dtype)

        expected_weight = (q_cf * weight_num).sum(dim=-1, keepdim=True)
        support = (
            expected_weight / cf_valid_count.clamp_min(1.0)
        ).clamp(0.0, 1.0)
        confidence = (certainty * support).clamp(0.0, 1.0)

        gain = gain.permute(0, 3, 1, 2).contiguous()
        confidence = confidence.permute(0, 3, 1, 2).contiguous()

        return observed, gain, confidence, (q_cf if need_q else None)


class FastScaleVectorized(_FastScaleBase):

    def __init__(self, src, hw, *, fuse_bn: bool, fold_output_bn: bool):
        super().__init__(
            src, hw,
            fuse_bn=fuse_bn,
            fold_output_bn=fold_output_bn,
        )
        self.token_norm = copy.deepcopy(src.token_norm).eval()

    def forward(self, x, posterior_temperature: float, relation_floor: float, need_q: bool):
        _, appearance_logits, appearance_prob = self._embed_and_appearance(x)
        b, n, k = appearance_prob.shape
        if n != self.n:
            raise ValueError("Unexpected token count")

        app = appearance_prob
        center = app.unsqueeze(1)  

        
        neigh = app[:, self.neigh_idx, :]   
        prev = app[:, self.prev_idx, :]     

        valid_out = self.valid_out.view(1, len(_DIRECTIONS), n, 1)
        valid_in = self.valid_in.view(1, len(_DIRECTIONS), n, 1)
        valid_out_f = valid_out.to(dtype=app.dtype)
        valid_in_f = valid_in.to(dtype=app.dtype)

        gates = self.gates_fast.to(dtype=app.dtype)
        gate4 = gates.view(1, len(_DIRECTIONS), 1, k)

        relation_prob = self.relation_prob_fast.to(dtype=app.dtype)
        row_max = self.row_max_fast.to(dtype=app.dtype)
        trans_norm = self.trans_norm_fast.to(dtype=app.dtype)

        
        compat_raw = torch.matmul(
            neigh, relation_prob.transpose(-1, -2)
        )
        compat_obs = (
            compat_raw / row_max.view(1, len(_DIRECTIONS), 1, k)
        ).clamp(0.0, 1.0)

        corr = (
            compat_obs.clamp_min(self.log_floor).log()
            * gate4
            * valid_out_f
        )
        rel_flat = corr.sum(dim=1)
        obs_count = self.valid_obs_count.view(1, n, 1).to(dtype=app.dtype)
        rel_flat = rel_flat / obs_count

        expected_gate = (center * gate4).sum(dim=-1, keepdim=True)
        expected_gate = expected_gate * valid_out_f
        expected_gate_compat = (
            center * gate4 * compat_obs
        ).sum(dim=-1, keepdim=True) * valid_out_f

        strength_num = expected_gate.sum(dim=1).squeeze(-1)
        consistency_num = expected_gate_compat.sum(dim=1).squeeze(-1)
        reliability_den = strength_num
        obs_count2 = self.valid_obs_count.view(1, n).to(dtype=app.dtype)
        structure_strength = (strength_num / obs_count2).clamp(0.0, 1.0)
        relation_consistency = torch.where(
            reliability_den > _EPS,
            consistency_num / reliability_den.clamp_min(_EPS),
            torch.ones_like(reliability_den),
        ).clamp(0.0, 1.0)

        observed = self._finish_projection(
            appearance_logits,
            appearance_prob,
            rel_flat,
            structure_strength,
            relation_consistency,
        )

        floor = max(float(relation_floor), _EPS)

        
        compat_fwd = compat_obs.clamp(floor, 1.0)
        fwd_weight = gate4 * valid_out_f
        fwd_energy = fwd_weight * (-compat_fwd.log())

        
        weighted_prev = prev * gate4
        reverse_rel = weighted_prev.sum(dim=-1, keepdim=True)
        reverse_num = torch.matmul(weighted_prev, trans_norm)
        compat_rev = (
            reverse_num / reverse_rel.clamp_min(_EPS)
        ).clamp(floor, 1.0)
        rev_weight = reverse_rel * valid_in_f
        rev_energy = rev_weight * (-compat_rev.log())

        energy_num = fwd_energy.sum(dim=1) + rev_energy.sum(dim=1)
        weight_num = fwd_weight.sum(dim=1) + rev_weight.sum(dim=1)

        candidate_energy = torch.where(
            weight_num > _EPS,
            energy_num / weight_num.clamp_min(_EPS),
            torch.zeros_like(energy_num),
        )

        q_cf = F.softmax(
            -candidate_energy.float() / float(posterior_temperature),
            dim=-1,
        ).to(dtype=app.dtype)

        e_obs = (app * candidate_energy).sum(dim=-1, keepdim=True)
        e_cf = (q_cf * candidate_energy).sum(dim=-1, keepdim=True)
        gain = e_obs - e_cf

        qf = q_cf.float().clamp_min(_EPS)
        entropy = -(qf * qf.log()).sum(dim=-1, keepdim=True)
        certainty = (
            1.0 - entropy / self.log_num_prototypes
        ).clamp(0.0, 1.0).to(dtype=app.dtype)

        expected_weight = (q_cf * weight_num).sum(dim=-1, keepdim=True)
        cf_count = self.valid_cf_count.view(1, n, 1).to(dtype=app.dtype)
        support = (expected_weight / cf_count).clamp(0.0, 1.0)
        confidence = (certainty * support).clamp(0.0, 1.0)

        gain = gain.transpose(1, 2).contiguous().view(b, 1, self.h, self.w)
        confidence = confidence.transpose(1, 2).contiguous().view(
            b, 1, self.h, self.w
        )

        if need_q:
            q_out = q_cf.view(b, self.h, self.w, k)
        else:
            q_out = None

        return observed, gain, confidence, q_out


class FastStructuralPrototypeBottleneckV3(nn.Module):








    def __init__(
        self,
        src: nn.Module,
        feature_hws: Sequence[Tuple[int, int]],
        *,
        backend: str,
        fuse_bn: bool,
        fold_output_bn: bool,
    ) -> None:
        super().__init__()
        if backend not in ("loop", "vectorized"):
            raise ValueError("backend must be 'loop' or 'vectorized'")
        if len(feature_hws) != 3:
            raise ValueError("feature_hws must have length 3")

        scale_cls = FastScaleLoop if backend == "loop" else FastScaleVectorized
        self.scales = nn.ModuleList([
            scale_cls(
                s, feature_hws[i],
                fuse_bn=fuse_bn,
                fold_output_bn=fold_output_bn,
            )
            for i, s in enumerate(src.scales)
        ])

        post = _make_post_modules(src, fuse_bn=fuse_bn)
        self.fine_down1 = post["fine_down1"]
        self.fine_down2 = post["fine_down2"]
        self.mid_down = post["mid_down"]
        self.fuse = post["fuse"]
        self.mixers = post["mixers"]
        self.to_8 = post["to_8"]
        self.decoder_proj = post["decoder_proj"]

        self.register_buffer(
            "cf_scale_weights_fast",
            src.cf_scale_weights.detach().clone().float(),
            persistent=False,
        )
        self.cf_posterior_temperature = float(src.cf_posterior_temperature)
        self.cf_relation_floor = float(src.cf_relation_floor)
        self.cf_gain_margin = float(src.cf_gain_margin)
        self.cf_gain_temperature = float(src.cf_gain_temperature)
        self.cf_max_intervention = float(src.cf_max_intervention)
        self.cf_detach_gate = bool(src.cf_detach_gate)
        self.backend = str(backend)
        self.fuse_bn = bool(fuse_bn)
        self.fold_output_bn = bool(
            fold_output_bn and all(s.output_folded for s in self.scales)
        )
        self._fast_inference_only = True

    def forward(self, features: Sequence[torch.Tensor], return_aux: bool = False):
        if return_aux:
            raise RuntimeError(
                "FastStructuralPrototypeBottleneckV3 is inference-only and does not produce SPRCRD aux. "
                "Use the original bottleneck for return_aux=True / Module-3 evaluation."
            )
        if len(features) != 3:
            raise ValueError("expected exactly three teacher feature tensors")

        fine_obs, g1, c1, _ = self.scales[0](
            features[0],
            self.cf_posterior_temperature,
            self.cf_relation_floor,
            False,
        )
        mid_obs, g2, c2, _ = self.scales[1](
            features[1],
            self.cf_posterior_temperature,
            self.cf_relation_floor,
            False,
        )
        coarse_obs, g3, c3, q3 = self.scales[2](
            features[2],
            self.cf_posterior_temperature,
            self.cf_relation_floor,
            True,
        )

        fine_scale = self.scales[0]
        mid_scale = self.scales[1]
        fine = fine_obs if fine_scale.output_folded else fine_scale.output_norm(fine_obs)
        mid = mid_obs if mid_scale.output_folded else mid_scale.output_norm(mid_obs)

        
        
        coarse_scale = self.scales[2]
        if coarse_scale.output_folded:
            coarse_basis = coarse_scale.proto_out
        else:
            coarse_basis = coarse_scale.proto_n
        if coarse_basis.dtype != q3.dtype:
            coarse_basis = coarse_basis.to(dtype=q3.dtype)

        coarse_cf = torch.matmul(q3, coarse_basis)
        coarse_cf = coarse_cf.permute(0, 3, 1, 2).contiguous()

        
        
        
        coarse_hw = coarse_obs.shape[-2:]
        if g1.shape[-2:] == (coarse_hw[0] * 4, coarse_hw[1] * 4):
            g1a = F.avg_pool2d(g1, kernel_size=4, stride=4)
            c1a = F.avg_pool2d(c1, kernel_size=4, stride=4)
        else:
            g1a = F.adaptive_avg_pool2d(g1, coarse_hw)
            c1a = F.adaptive_avg_pool2d(c1, coarse_hw)

        if g2.shape[-2:] == (coarse_hw[0] * 2, coarse_hw[1] * 2):
            g2a = F.avg_pool2d(g2, kernel_size=2, stride=2)
            c2a = F.avg_pool2d(c2, kernel_size=2, stride=2)
        else:
            g2a = F.adaptive_avg_pool2d(g2, coarse_hw)
            c2a = F.adaptive_avg_pool2d(c2, coarse_hw)

        weights = self.cf_scale_weights_fast.to(
            device=g3.device, dtype=g3.dtype
        )
        hier_gain = weights[0] * g1a + weights[1] * g2a + weights[2] * g3
        hier_conf = weights[0] * c1a + weights[1] * c2a + weights[2] * c3

        verified_conf = (c3 * hier_conf).sqrt().clamp(0.0, 1.0)
        excess = F.relu(hier_gain - self.cf_gain_margin)
        gain_activation = 1.0 - torch.exp(
            -excess / self.cf_gain_temperature
        )
        alpha = float(self.scales[2].relation_alpha)
        gate = (
            alpha
            * self.cf_max_intervention
            * verified_conf
            * gain_activation
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
        return decoder_input, {}

__all__ = ["FastStructuralPrototypeBottleneckV3", "_FastScaleBase", "_tensor_error", "_passes"]
