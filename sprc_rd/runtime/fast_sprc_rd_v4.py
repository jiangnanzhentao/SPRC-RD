


































from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fast_sprc_rd_v3 as v3


_EPS = 1e-6
_DIRECTIONS = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)


class FastScaleVectorizedV4(v3._FastScaleBase):









    def __init__(
        self,
        src: nn.Module,
        hw: Tuple[int, int],
        *,
        fuse_bn: bool,
        fold_output_bn: bool,
        sparse_backend: str,
    ) -> None:
        super().__init__(
            src,
            hw,
            fuse_bn=fuse_bn,
            fold_output_bn=fold_output_bn,
        )
        if sparse_backend not in ("sort_i64", "drop_i64", "drop_i32"):
            raise ValueError("invalid sparse_backend")
        self.sparse_backend = str(sparse_backend)
        self.token_norm = v3.copy.deepcopy(src.token_norm).eval()

        
        
        
        relation_prob = self.relation_prob_fast.detach().float()
        row_max = self.row_max_fast.detach().float()
        gates = self.gates_fast.detach().float()
        trans_norm = self.trans_norm_fast.detach().float()

        
        
        
        relation_scaled = relation_prob / row_max.unsqueeze(-1)

        
        
        
        gate_transition = gates.unsqueeze(-1) * trans_norm

        
        
        gate_out = (
            gates[:, None, :]
            * self.valid_out.detach().float()[:, :, None]
        )
        fwd_weight_sum = gate_out.sum(dim=0)  

        obs_inv = self.valid_obs_count.detach().float().reciprocal()
        cf_inv = self.valid_cf_count.detach().float().reciprocal()

        self.register_buffer(
            "relation_scaled_fast", relation_scaled, persistent=False
        )
        self.register_buffer(
            "gate_transition_fast", gate_transition, persistent=False
        )
        self.register_buffer(
            "gate_column_fast", gates.unsqueeze(-1), persistent=False
        )
        self.register_buffer(
            "gate_out_fast", gate_out, persistent=False
        )
        self.register_buffer(
            "fwd_weight_sum_fast", fwd_weight_sum, persistent=False
        )
        self.register_buffer(
            "obs_inv_fast", obs_inv, persistent=False
        )
        self.register_buffer(
            "cf_inv_fast", cf_inv, persistent=False
        )
        self.register_buffer(
            "valid_in_float_fast",
            self.valid_in.detach().float(),
            persistent=False,
        )

        
        
        
        k = self.num_prototypes
        max_drop = max(0, k - self.min_topk)
        self.max_drop = int(max_drop)

        key_dtype = (
            torch.int32 if self.sparse_backend == "drop_i32"
            else torch.int64
        )
        ids = torch.arange(k, device=relation_prob.device, dtype=key_dtype)
        tie_break = (k - ids)
        self.register_buffer(
            "sparse_tie_break", tie_break, persistent=False
        )
        self.register_buffer(
            "sparse_rank_values",
            torch.arange(k, device=relation_prob.device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "drop_rank_values",
            torch.arange(
                max_drop,
                device=relation_prob.device,
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _deterministic_key(self, prob: torch.Tensor) -> torch.Tensor:
        k = self.num_prototypes
        if self.sparse_backend == "drop_i32":
            
            q = torch.round(
                prob.detach().float() * float(self.quant_scale)
            ).to(torch.int32)
            tie = self.sparse_tie_break
            return q * int(k + 1) + tie

        q = torch.round(
            prob.detach().float() * float(self.quant_scale)
        ).to(torch.int64)
        tie = self.sparse_tie_break
        return q * int(k + 1) + tie

    def _sparse_projection(
        self,
        prob: torch.Tensor,
        suspicion: torch.Tensor,
    ) -> torch.Tensor:
        k = self.num_prototypes
        if self.relation_alpha <= 0.0 or self.min_topk >= k:
            return prob

        k_float = (
            k
            - self.relation_alpha
            * suspicion
            * float(k - self.min_topk)
        )
        keep_count = (
            torch.round(k_float)
            .to(torch.long)
            .clamp_(self.min_topk, k)
        )

        key = self._deterministic_key(prob)

        if self.sparse_backend == "sort_i64":
            
            order = torch.argsort(key, dim=-1, descending=True)
            ranks = torch.empty_like(order)
            rank_values = self.sparse_rank_values
            rank_values = rank_values.view(
                *([1] * (order.dim() - 1)), k
            ).expand_as(order)
            ranks.scatter_(-1, order, rank_values)
            mask = ranks < keep_count.unsqueeze(-1)
            sparse = prob * mask.to(dtype=prob.dtype)
            return sparse / sparse.sum(
                dim=-1, keepdim=True
            ).clamp_min(_EPS)

        
        
        drop_count = k - keep_count

        
        
        low_idx = torch.topk(
            key,
            k=self.max_drop,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
        to_drop = (
            self.drop_rank_values.view(
                *([1] * (low_idx.dim() - 1)),
                self.max_drop,
            )
            < drop_count.unsqueeze(-1)
        )

        drop_mask = torch.zeros_like(prob, dtype=torch.bool)
        drop_mask.scatter_(-1, low_idx, to_drop)
        sparse = prob.masked_fill(drop_mask, 0.0)
        return sparse / sparse.sum(
            dim=-1, keepdim=True
        ).clamp_min(_EPS)

    def forward(
        self,
        x: torch.Tensor,
        posterior_temperature: float,
        relation_floor: float,
        need_q: bool,
    ):
        _, appearance_logits, appearance_prob = self._embed_and_appearance(x)
        b, n, k = appearance_prob.shape
        if n != self.n:
            raise ValueError(
                "Fast v4 scale prepared for N={}, got N={}".format(
                    self.n, n
                )
            )

        app = appearance_prob
        center = app.unsqueeze(1)  

        
        neigh = app[:, self.neigh_idx, :]   
        prev = app[:, self.prev_idx, :]     

        relation_scaled = self.relation_scaled_fast.to(dtype=app.dtype)
        gate_transition = self.gate_transition_fast.to(dtype=app.dtype)
        gate_column = self.gate_column_fast.to(dtype=app.dtype)
        gate_out = self.gate_out_fast.to(dtype=app.dtype)
        fwd_weight_sum = self.fwd_weight_sum_fast.to(dtype=app.dtype)
        obs_inv = self.obs_inv_fast.to(dtype=app.dtype)
        cf_inv = self.cf_inv_fast.to(dtype=app.dtype)
        valid_in_f = self.valid_in_float_fast.to(dtype=app.dtype)

        
        
        
        
        
        
        
        compat_obs = torch.matmul(
            neigh,
            relation_scaled.transpose(-1, -2),
        ).clamp(0.0, 1.0)

        
        log_compat = compat_obs.clamp_min(self.log_floor).log()
        rel_flat = (log_compat * gate_out.unsqueeze(0)).sum(dim=1)
        rel_flat = rel_flat * obs_inv.view(1, n, 1)

        expected_gate = (
            center * gate_out.unsqueeze(0)
        ).sum(dim=-1)
        expected_gate_compat = (
            center
            * gate_out.unsqueeze(0)
            * compat_obs
        ).sum(dim=-1)

        strength_num = expected_gate.sum(dim=1)
        consistency_num = expected_gate_compat.sum(dim=1)
        reliability_den = strength_num

        structure_strength = (
            strength_num * obs_inv.view(1, n)
        ).clamp(0.0, 1.0)
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
        fwd_energy_sum = (
            gate_out.unsqueeze(0) * (-compat_fwd.log())
        ).sum(dim=1)

        
        
        reverse_rel = torch.matmul(
            prev,
            gate_column,
        )
        
        reverse_num = torch.matmul(
            prev,
            gate_transition,
        )
        compat_rev = (
            reverse_num / reverse_rel.clamp_min(_EPS)
        ).clamp(floor, 1.0)

        rev_weight = (
            reverse_rel
            * valid_in_f.view(
                1, len(_DIRECTIONS), n, 1
            )
        )
        rev_energy_sum = (
            rev_weight * (-compat_rev.log())
        ).sum(dim=1)

        energy_num = fwd_energy_sum + rev_energy_sum
        weight_num = (
            fwd_weight_sum.unsqueeze(0)
            + rev_weight.sum(dim=1)
        )

        candidate_energy = torch.where(
            weight_num > _EPS,
            energy_num / weight_num.clamp_min(_EPS),
            torch.zeros_like(energy_num),
        )

        if candidate_energy.dtype == torch.float32:
            q_cf = F.softmax(
                -candidate_energy / float(posterior_temperature),
                dim=-1,
            )
        else:
            q_cf = F.softmax(
                -candidate_energy.float()
                / float(posterior_temperature),
                dim=-1,
            ).to(dtype=app.dtype)

        e_obs = (app * candidate_energy).sum(
            dim=-1, keepdim=True
        )
        e_cf = (q_cf * candidate_energy).sum(
            dim=-1, keepdim=True
        )
        gain = e_obs - e_cf

        if q_cf.dtype == torch.float32:
            qf = q_cf.clamp_min(_EPS)
        else:
            qf = q_cf.float().clamp_min(_EPS)
        entropy = -(qf * qf.log()).sum(
            dim=-1, keepdim=True
        )
        certainty = (
            1.0 - entropy / self.log_num_prototypes
        ).clamp(0.0, 1.0)
        if certainty.dtype != app.dtype:
            certainty = certainty.to(dtype=app.dtype)

        expected_weight = (
            q_cf * weight_num
        ).sum(dim=-1, keepdim=True)
        support = (
            expected_weight
            * cf_inv.view(1, n, 1)
        ).clamp(0.0, 1.0)
        confidence = (
            certainty * support
        ).clamp(0.0, 1.0)

        gain = gain.transpose(1, 2).contiguous().view(
            b, 1, self.h, self.w
        )
        confidence = confidence.transpose(
            1, 2
        ).contiguous().view(
            b, 1, self.h, self.w
        )

        if need_q:
            q_out = q_cf.view(
                b, self.h, self.w, k
            )
        else:
            q_out = None

        return observed, gain, confidence, q_out


class FastStructuralPrototypeBottleneckV4(v3.FastStructuralPrototypeBottleneckV3):




    def __init__(
        self,
        src: nn.Module,
        feature_hws: Sequence[Tuple[int, int]],
        *,
        fuse_bn: bool,
        fold_output_bn: bool,
        sparse_backend: str,
    ) -> None:
        
        super().__init__(
            src,
            feature_hws,
            backend="vectorized",
            fuse_bn=fuse_bn,
            fold_output_bn=fold_output_bn,
        )

        
        self.scales = nn.ModuleList([
            FastScaleVectorizedV4(
                s,
                feature_hws[i],
                fuse_bn=fuse_bn,
                fold_output_bn=fold_output_bn,
                sparse_backend=sparse_backend,
            )
            for i, s in enumerate(src.scales)
        ])
        self.backend = "v4_" + str(sparse_backend)
        self.sparse_backend = str(sparse_backend)
        self.fold_output_bn = bool(
            fold_output_bn
            and all(s.output_folded for s in self.scales)
        )

__all__ = ["FastScaleVectorizedV4", "FastStructuralPrototypeBottleneckV4"]
