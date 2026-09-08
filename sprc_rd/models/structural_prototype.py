


































from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    _DIRECTIONS,
    _EPS,
    CompactSpatialMixer,
    ConvNormAct,
    DepthwiseDownsample,
    StructuralPrototypeProjection,
    _make_norm_2d,
)


class RelationGuidedPrototypeIntervention(StructuralPrototypeProjection):







    def normalized_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                "Expected [B,{},H,W], got {}".format(self.in_channels, tuple(x.shape))
            )
        z_map = self.adapter(x)
        b, d, h, w = z_map.shape
        tokens = z_map.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_norm(tokens)
        tokens_n = F.normalize(tokens.float(), dim=-1, eps=_EPS).to(dtype=tokens.dtype)
        return tokens_n.transpose(1, 2).contiguous().view(b, d, h, w)

    def appearance_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:

        if embedding.dim() != 4 or embedding.size(1) != self.embed_dim:
            raise ValueError(
                "Expected embedding [B,{},H,W], got {}".format(
                    self.embed_dim, tuple(embedding.shape)
                )
            )
        tokens_n = F.normalize(embedding.float(), dim=1, eps=_EPS)
        tokens_n = tokens_n.flatten(2).transpose(1, 2).contiguous()
        proto_n = F.normalize(self.prototypes.float(), dim=-1, eps=_EPS)
        logits = torch.matmul(tokens_n, proto_n.t()) / self.temperature
        b, _, k = logits.shape
        h, w = embedding.shape[-2:]
        return F.softmax(logits, dim=-1).view(b, h, w, k)

    @torch.no_grad()
    def clean_appearance_target(self, x: torch.Tensor) -> torch.Tensor:

        return self.appearance_from_embedding(self.normalized_embedding(x)).detach()

    def _observed_projection_pre_norm(
        self,
        embedding: torch.Tensor,
        return_aux: bool = True,
    ):





        b, d, h, w = embedding.shape
        tokens_n = F.normalize(embedding.float(), dim=1, eps=_EPS)
        tokens_n = tokens_n.flatten(2).transpose(1, 2).contiguous().to(dtype=embedding.dtype)
        proto_n = F.normalize(self.prototypes.float(), dim=-1, eps=_EPS).to(dtype=tokens_n.dtype)

        appearance_logits = torch.matmul(tokens_n, proto_n.t()) / self.temperature
        appearance_prob = F.softmax(appearance_logits, dim=-1)

        rel_corr, structure_strength, relation_consistency = self._relation_evidence(
            appearance_prob, h, w
        )
        alpha = self.relation_alpha
        final_logits = appearance_logits + (
            float(alpha) * self.relation_weight * rel_corr.view(b, h * w, self.num_prototypes)
        )
        final_prob = F.softmax(final_logits, dim=-1)

        suspicion_map = structure_strength * (1.0 - relation_consistency)
        suspicion = suspicion_map.view(b, h * w)
        sparse_prob, topk = self._adaptive_sparse_projection(final_prob, suspicion, alpha)

        normal_tokens = torch.matmul(sparse_prob, proto_n)
        pre_norm = normal_tokens.transpose(1, 2).contiguous().view(b, d, h, w)

        if not return_aux:
            return pre_norm, {}, appearance_prob.view(b, h, w, self.num_prototypes)

        usage = sparse_prob.mean(dim=(0, 1)).clamp_min(_EPS)
        usage = usage / usage.sum()
        proto_k_eff = (-(usage * usage.log()).sum()).exp()
        p = appearance_prob.clamp_min(_EPS)
        token_entropy = -(p * p.log()).sum(dim=-1).mean()
        token_k_eff = token_entropy.exp()
        relation_gate = self.relation_gate().to(device=tokens_n.device)
        aux = {
            "appearance_prob": appearance_prob,
            "sparse_prob": sparse_prob,
            "proto_k_eff": proto_k_eff.detach(),
            "token_attn_k_eff": token_k_eff.detach(),
            "token_attn_entropy": token_entropy.detach(),
            "avg_topk": topk.float().mean().detach(),
            "min_topk": topk.min().detach(),
            "max_topk": topk.max().detach(),
            "structure_strength": structure_strength.mean().detach(),
            "relation_consistency": relation_consistency.mean().detach(),
            "suspicion": suspicion.mean().detach(),
            "relation_gate_mean": relation_gate.mean().detach(),
            "relation_gate_max": relation_gate.max().detach(),
            
            
            
            "relation_alpha": tokens_n.new_full((), float(alpha)).detach(),
        }
        return pre_norm, aux, appearance_prob.view(b, h, w, self.num_prototypes)

    def forward_pre_norm(self, x: torch.Tensor, return_aux: bool = True):
        embedding = self.normalized_embedding(x)
        pre_norm, aux, appearance_map = self._observed_projection_pre_norm(
            embedding, return_aux=return_aux
        )
        return pre_norm, aux, embedding, appearance_map

    def counterfactual_reasoning(
        self,
        appearance_map: torch.Tensor,
        posterior_temperature: float = 0.35,
        relation_floor: float = 1e-4,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:































        if appearance_map.dim() != 4 or appearance_map.size(-1) != self.num_prototypes:
            raise ValueError(
                "Expected appearance_map [B,H,W,{}], got {}".format(
                    self.num_prototypes, tuple(appearance_map.shape)
                )
            )
        if posterior_temperature <= 0:
            raise ValueError("posterior_temperature must be > 0")

        app = appearance_map
        b, h, w, k = app.shape
        dtype = app.dtype
        device = app.device

        trans_all = self.relation_prob.to(device=device, dtype=dtype)
        row_gate_all = self.relation_gate().to(device=device, dtype=dtype)

        
        
        
        energy_num = app.new_zeros((b, h, w, k))
        weight_num = app.new_zeros((b, h, w, k))
        valid_count = app.new_zeros((1, h, w, 1))

        floor = max(float(relation_floor), _EPS)

        for d, (dy, dx) in enumerate(_DIRECTIONS):
            center_slice, neigh_slice = self._direction_slices(h, w, dy, dx)
            cy, cx = center_slice
            ny, nx = neigh_slice

            left_app = app[:, cy, cx, :]   
            right_app = app[:, ny, nx, :]  
            trans = trans_all[d]           
            row_gate = row_gate_all[d]     
            row_max = trans.max(dim=-1).values.clamp_min(_EPS)
            trans_norm = (trans / row_max[:, None]).clamp(0.0, 1.0)

            
            
            compat_fwd = torch.matmul(right_app, trans_norm.t()).clamp(floor, 1.0)
            rel_fwd = row_gate.view(1, 1, 1, k).expand_as(compat_fwd)
            nll_fwd = -compat_fwd.log()
            energy_num[:, cy, cx, :] += rel_fwd * nll_fwd
            weight_num[:, cy, cx, :] += rel_fwd
            valid_count[:, cy, cx, :] += 1.0

            
            
            weighted_left = left_app * row_gate.view(1, 1, 1, k)
            reverse_rel = weighted_left.sum(dim=-1, keepdim=True)  
            reverse_num = torch.matmul(weighted_left, trans_norm)  
            compat_rev = (reverse_num / reverse_rel.clamp_min(_EPS)).clamp(floor, 1.0)
            nll_rev = -compat_rev.log()
            rel_rev = reverse_rel.expand_as(compat_rev)
            energy_num[:, ny, nx, :] += rel_rev * nll_rev
            weight_num[:, ny, nx, :] += rel_rev
            valid_count[:, ny, nx, :] += 1.0

        candidate_energy = torch.where(
            weight_num > _EPS,
            energy_num / weight_num.clamp_min(_EPS),
            torch.zeros_like(energy_num),
        )

        
        
        cf_logits = -candidate_energy.float() / float(posterior_temperature)
        q_cf = F.softmax(cf_logits, dim=-1).to(dtype=dtype)

        p_obs = app
        e_obs = (p_obs * candidate_energy).sum(dim=-1, keepdim=True)
        e_cf = (q_cf * candidate_energy).sum(dim=-1, keepdim=True)
        gain = e_obs - e_cf

        qf = q_cf.float().clamp_min(_EPS)
        entropy = -(qf * qf.log()).sum(dim=-1, keepdim=True)
        certainty = (1.0 - entropy / math.log(float(k))).clamp(0.0, 1.0).to(dtype=dtype)

        
        
        expected_weight = (q_cf * weight_num).sum(dim=-1, keepdim=True)
        support = expected_weight / valid_count.clamp_min(1.0)
        support = support.clamp(0.0, 1.0)
        confidence = (certainty * support).clamp(0.0, 1.0)

        def nchw(x: torch.Tensor) -> torch.Tensor:
            return x.permute(0, 3, 1, 2).contiguous()

        result = {
            "q_cf": q_cf,
            "energy_gain": nchw(gain),
            "confidence": nchw(confidence),
        }
        if return_aux:
            disagreement = 0.5 * (p_obs - q_cf).abs().sum(dim=-1, keepdim=True)
            result.update(
                {
                    "candidate_energy": candidate_energy,
                    "observed_energy": nchw(e_obs),
                    "counterfactual_energy": nchw(e_cf),
                    "posterior_certainty": nchw(certainty),
                    "relation_support": nchw(support),
                    "disagreement": nchw(disagreement),
                }
            )
        return result

    def counterfactual_pre_norm(self, q_cf: torch.Tensor) -> torch.Tensor:

        if q_cf.dim() != 4 or q_cf.size(-1) != self.num_prototypes:
            raise ValueError("q_cf has invalid shape: {}".format(tuple(q_cf.shape)))
        proto_n = F.normalize(self.prototypes.float(), dim=-1, eps=_EPS).to(dtype=q_cf.dtype)
        tokens = torch.matmul(q_cf, proto_n)  
        return tokens.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor, return_aux: bool = True):
        pre_norm, aux, _, _ = self.forward_pre_norm(x, return_aux=return_aux)
        return self.output_norm(pre_norm), aux


class StructuralPrototypeBottleneck(nn.Module):








    def __init__(
        self,
        feature_channels: Sequence[int],
        embed_dims: Sequence[int] = (128, 160, 256),
        num_prototypes: Sequence[int] = (4, 4, 4),
        min_topk: Sequence[int] = (4, 2, 1),
        relation_weights: Sequence[float] = (0.15, 0.30, 0.45),
        temperature: float = 0.2,
        fusion_dim: int = 256,
        decoder_in_channels: int = 2048,
        fusion_blocks: int = 2,
        relation_warmup_epochs: int = 10,
        relation_ramp_epochs: int = 10,
        relation_momentum: float = 0.5,
        relation_smoothing: float = 1.0,
        norm: str = "bn",
        quant_scale: float = 1e6,
        
        cf_posterior_temperature: float = 0.35,
        cf_relation_floor: float = 1e-4,
        cf_scale_weights: Sequence[float] = (0.15, 0.25, 0.60),
        cf_gain_margin: float = 0.08,
        cf_gain_temperature: float = 0.12,
        cf_max_intervention: float = 1.0,
        cf_detach_gate: bool = True,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 3 or len(embed_dims) != 3:
            raise ValueError("feature_channels and embed_dims must have length 3")
        if len(num_prototypes) != 3 or len(min_topk) != 3 or len(relation_weights) != 3:
            raise ValueError("num_prototypes/min_topk/relation_weights must have length 3")
        if len(cf_scale_weights) != 3:
            raise ValueError("cf_scale_weights must have length 3")
        if cf_posterior_temperature <= 0 or cf_gain_temperature <= 0:
            raise ValueError("counterfactual temperatures must be > 0")
        if not (0.0 < cf_max_intervention <= 1.0):
            raise ValueError("cf_max_intervention must be in (0,1]")

        self.feature_channels = tuple(int(x) for x in feature_channels)
        self.embed_dims = tuple(int(x) for x in embed_dims)
        self.num_prototypes = tuple(int(x) for x in num_prototypes)
        self.min_topk = tuple(int(x) for x in min_topk)
        self.relation_weights = tuple(float(x) for x in relation_weights)
        self.fusion_dim = int(fusion_dim)
        self.decoder_in_channels = int(decoder_in_channels)

        self.cf_posterior_temperature = float(cf_posterior_temperature)
        self.cf_relation_floor = float(cf_relation_floor)
        sw = torch.tensor([float(x) for x in cf_scale_weights], dtype=torch.float32)
        if torch.any(sw < 0) or float(sw.sum()) <= 0:
            raise ValueError("cf_scale_weights must be nonnegative with positive sum")
        sw = sw / sw.sum()
        self.register_buffer("cf_scale_weights", sw)
        self.cf_gain_margin = float(cf_gain_margin)
        self.cf_gain_temperature = float(cf_gain_temperature)
        self.cf_max_intervention = float(cf_max_intervention)
        self.cf_detach_gate = bool(cf_detach_gate)

        scales: List[nn.Module] = []
        for c, d, k, mk, rw in zip(
            self.feature_channels,
            self.embed_dims,
            self.num_prototypes,
            self.min_topk,
            self.relation_weights,
        ):
            scales.append(
                RelationGuidedPrototypeIntervention(
                    in_channels=c,
                    embed_dim=d,
                    num_prototypes=k,
                    temperature=temperature,
                    min_topk=mk,
                    relation_weight=rw,
                    relation_warmup_epochs=relation_warmup_epochs,
                    relation_ramp_epochs=relation_ramp_epochs,
                    relation_momentum=relation_momentum,
                    relation_smoothing=relation_smoothing,
                    norm=norm,
                    quant_scale=quant_scale,
                )
            )
        self.scales = nn.ModuleList(scales)

        d1, d2, d3 = self.embed_dims
        self.fine_down1 = DepthwiseDownsample(d1, d1, norm=norm)
        self.fine_down2 = DepthwiseDownsample(d1, d1, norm=norm)
        self.mid_down = DepthwiseDownsample(d2, d2, norm=norm)
        self.fuse = ConvNormAct(d1 + d2 + d3, fusion_dim, kernel_size=1, norm=norm, act=True)
        self.mixers = nn.Sequential(
            *[
                CompactSpatialMixer(fusion_dim, expansion=2.0, norm=norm)
                for _ in range(int(fusion_blocks))
            ]
        )
        self.to_8 = DepthwiseDownsample(fusion_dim, fusion_dim, norm=norm)
        self.decoder_proj = nn.Sequential(
            nn.Conv2d(fusion_dim, decoder_in_channels, kernel_size=1, bias=False),
            _make_norm_2d(decoder_in_channels, norm),
        )
        self._reset_projection_parameters()

    def _reset_projection_parameters(self) -> None:
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _check_spatial_hierarchy(features: Sequence[torch.Tensor]) -> None:
        if len(features) != 3:
            raise ValueError("RD encoder must return exactly three features")
        h1, w1 = features[0].shape[-2:]
        h2, w2 = features[1].shape[-2:]
        h3, w3 = features[2].shape[-2:]
        if not (h1 == 2 * h2 == 4 * h3 and w1 == 2 * w2 == 4 * w3):
            raise ValueError(
                "Expected x2 hierarchy F1/F2/F3, got {} / {} / {}".format(
                    features[0].shape[-2:], features[1].shape[-2:], features[2].shape[-2:]
                )
            )

    def begin_epoch(self, epoch: int) -> None:
        for scale in self.scales:
            scale.begin_epoch(epoch)

    @torch.no_grad()
    def finalize_relation_epoch(self) -> List[bool]:
        return [scale.finalize_relation_epoch() for scale in self.scales]

    @torch.no_grad()
    def extract_clean_coarse_appearance(self, coarse_teacher_feature: torch.Tensor) -> torch.Tensor:
        scale = self.scales[2]
        assert isinstance(scale, RelationGuidedPrototypeIntervention)
        return scale.clean_appearance_target(coarse_teacher_feature)

    def _counterfactual_gate(
        self,
        cf_by_scale: Sequence[Dict[str, torch.Tensor]],
        coarse_hw: Tuple[int, int],
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        gains = []
        confidences = []
        disagreements = [] if return_aux else None
        for cf in cf_by_scale:
            g = cf["energy_gain"]
            c = cf["confidence"]
            if g.shape[-2:] != coarse_hw:
                g = F.adaptive_avg_pool2d(g, coarse_hw)
                c = F.adaptive_avg_pool2d(c, coarse_hw)
            gains.append(g)
            confidences.append(c)
            if return_aux:
                d = cf["disagreement"]
                if d.shape[-2:] != coarse_hw:
                    d = F.adaptive_avg_pool2d(d, coarse_hw)
                disagreements.append(d)

        weights = self.cf_scale_weights.to(device=gains[0].device, dtype=gains[0].dtype)
        hier_gain = sum(weights[i] * gains[i] for i in range(3))
        hier_conf = sum(weights[i] * confidences[i] for i in range(3))

        
        
        coarse_conf = confidences[2]
        verified_conf = (coarse_conf * hier_conf).sqrt().clamp(0.0, 1.0)

        
        excess = F.relu(hier_gain - self.cf_gain_margin)
        gain_activation = 1.0 - torch.exp(-excess / self.cf_gain_temperature)

        alpha = float(self.scales[2].relation_alpha)
        gate = float(alpha) * self.cf_max_intervention * verified_conf * gain_activation
        gate = gate.clamp(0.0, self.cf_max_intervention)
        if self.cf_detach_gate:
            gate = gate.detach()

        diag = {
            "hierarchical_gain": hier_gain,
            "hierarchical_confidence": hier_conf,
            "verified_confidence": verified_conf,
            "gain_activation": gain_activation,
        }
        if return_aux:
            diag["hierarchical_disagreement"] = sum(
                weights[i] * disagreements[i] for i in range(3)
            )
        return gate, diag

    def forward(self, features: Sequence[torch.Tensor], return_aux: bool = True):
        self._check_spatial_hierarchy(features)

        pre_norms: List[torch.Tensor] = []
        scale_aux: List[Dict[str, torch.Tensor]] = []
        cf_by_scale: List[Dict[str, torch.Tensor]] = []

        for feat, module in zip(features, self.scales):
            assert isinstance(module, RelationGuidedPrototypeIntervention)
            pre, aux, _embedding, app_map = module.forward_pre_norm(
                feat, return_aux=return_aux
            )
            cf = module.counterfactual_reasoning(
                app_map,
                posterior_temperature=self.cf_posterior_temperature,
                relation_floor=self.cf_relation_floor,
                return_aux=return_aux,
            )
            pre_norms.append(pre)
            scale_aux.append(aux)
            cf_by_scale.append(cf)

        fine_pre, mid_pre, coarse_pre_obs = pre_norms
        fine = self.scales[0].output_norm(fine_pre)
        mid = self.scales[1].output_norm(mid_pre)

        coarse_cf_pre = self.scales[2].counterfactual_pre_norm(cf_by_scale[2]["q_cf"])
        coarse_hw = tuple(coarse_pre_obs.shape[-2:])
        intervention_gate, gate_diag = self._counterfactual_gate(
            cf_by_scale, coarse_hw, return_aux=return_aux
        )
        coarse_pre = coarse_pre_obs + intervention_gate * (coarse_cf_pre - coarse_pre_obs)
        coarse = self.scales[2].output_norm(coarse_pre)

        fine_aligned = self.fine_down2(self.fine_down1(fine))
        mid_aligned = self.mid_down(mid)
        if fine_aligned.shape[-2:] != coarse.shape[-2:] or mid_aligned.shape[-2:] != coarse.shape[-2:]:
            raise RuntimeError("Scale alignment failed")

        fused = self.fuse(torch.cat([fine_aligned, mid_aligned, coarse], dim=1))
        fused = self.mixers(fused)
        fused = self.to_8(fused)
        decoder_input = self.decoder_proj(fused).contiguous()

        if not return_aux:
            return decoder_input, {}

        
        names = ("fine", "mid", "coarse")
        aux: Dict[str, object] = {}
        for name, sa_aux, cf in zip(names, scale_aux, cf_by_scale):
            merged = dict(sa_aux)
            merged.update(
                {
                    "cf_q": cf["q_cf"],
                    "cf_observed_energy": cf["observed_energy"],
                    "cf_counterfactual_energy": cf["counterfactual_energy"],
                    "cf_energy_gain": cf["energy_gain"],
                    "cf_confidence": cf["confidence"],
                    "cf_posterior_certainty": cf["posterior_certainty"],
                    "cf_relation_support": cf["relation_support"],
                    "cf_disagreement": cf["disagreement"],
                }
            )
            aux[name] = merged

        
        
        fused_h, fused_w = fused.shape[-2:]
        fused_shape = torch.arange(2, device=fused.device, dtype=torch.int64)
        fused_shape = fused_shape.mul(int(fused_w) - int(fused_h)).add(int(fused_h))

        aux.update(
            {
                "intervention_gate": intervention_gate,
                "expected_coarse_projection": coarse_cf_pre,
                "multiscale_structural_gain": gate_diag["hierarchical_gain"],
                "multiscale_relation_confidence": gate_diag["hierarchical_confidence"],
                "prototype_state_confidence": gate_diag["verified_confidence"],
                "prototype_state_disagreement": gate_diag["hierarchical_disagreement"],
                "intervention_activation": gate_diag["gain_activation"],
                "fused_shape": fused_shape,
            }
        )
        return decoder_input, aux

    @torch.no_grad()
    def accumulate_relation_counts(
        self,
        aux: Dict[str, Dict[str, torch.Tensor]],
        features: Sequence[torch.Tensor],
        clean_count: int = -1,
    ) -> None:

        for name, module, feat in zip(("fine", "mid", "coarse"), self.scales, features):
            if name not in aux or "appearance_prob" not in aux[name]:
                continue
            prob = aux[name]["appearance_prob"]
            if clean_count >= 0:
                prob = prob[:clean_count]
            if prob.size(0) == 0:
                continue
            h, w = feat.shape[-2:]
            module.accumulate_relation_counts(prob, h, w)

    def parameter_report(self) -> Dict[str, int]:
        all_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        
        
        return {
            "total_trainable": all_trainable,
            "fine": sum(p.numel() for p in self.scales[0].parameters() if p.requires_grad),
            "mid": sum(p.numel() for p in self.scales[1].parameters() if p.requires_grad),
            "coarse": sum(p.numel() for p in self.scales[2].parameters() if p.requires_grad),
            "sprc_rd_extra_trainable": 0,
            "fusion_and_decoder_interface": sum(
                p.numel()
                for name, p in self.named_parameters()
                if not name.startswith("scales.") and p.requires_grad
            ),
        }


__all__ = [
    "RelationGuidedPrototypeIntervention",
    "StructuralPrototypeBottleneck",
]
