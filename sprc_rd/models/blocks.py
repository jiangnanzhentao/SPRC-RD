


























from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-6
_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),   
    (1, 0),   
    (1, 1),   
    (1, -1),  
)


def _make_norm_2d(channels: int, norm: str = "bn") -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = min(32, channels)
        while channels % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError("norm must be 'bn', 'gn', or 'none', got {!r}".format(norm))


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        norm: str = "bn",
        act: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            _make_norm_2d(out_channels, norm),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class DepthwiseDownsample(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn") -> None:
        super().__init__()
        self.dw = ConvNormAct(
            in_channels, in_channels, kernel_size=3, stride=2,
            groups=in_channels, norm=norm, act=True,
        )
        self.pw = ConvNormAct(in_channels, out_channels, kernel_size=1, norm=norm, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class GRN2d(nn.Module):


    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class CompactSpatialMixer(nn.Module):


    def __init__(self, channels: int, expansion: float = 2.0, norm: str = "bn") -> None:
        super().__init__()
        hidden = max(channels, int(round(channels * expansion)))
        self.dw = ConvNormAct(
            channels, channels, kernel_size=5, stride=1,
            groups=channels, norm=norm, act=True,
        )
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN2d(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.out_norm = _make_norm_2d(channels, norm)

        
        if isinstance(self.out_norm, nn.BatchNorm2d):
            nn.init.zeros_(self.out_norm.weight)
            nn.init.zeros_(self.out_norm.bias)
        elif isinstance(self.out_norm, nn.GroupNorm):
            nn.init.zeros_(self.out_norm.weight)
            nn.init.zeros_(self.out_norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw1(y)
        y = self.act(y)
        y = self.grn(y)
        y = self.pw2(y)
        y = self.out_norm(y)
        return x + y


def _entropy(prob: torch.Tensor, dim: int = -1, eps: float = _EPS) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def _deterministic_order(prob: torch.Tensor, quant_scale: float = 1e6) -> torch.Tensor:






    p = prob.detach().float()
    k = p.shape[-1]
    q = torch.round(p * float(quant_scale)).to(torch.int64)
    ids = torch.arange(k, device=p.device, dtype=torch.int64)
    
    key = q * (k + 1) + (k - ids)
    return torch.argsort(key, dim=-1, descending=True)


class StructuralPrototypeProjection(nn.Module):


    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_prototypes: int = 32,
        temperature: float = 0.2,
        min_topk: int = 8,
        relation_weight: float = 0.3,
        relation_warmup_epochs: int = 10,
        relation_ramp_epochs: int = 10,
        relation_momentum: float = 0.5,
        relation_smoothing: float = 1.0,
        relation_log_floor: float = -4.0,
        norm: str = "bn",
        quant_scale: float = 1e6,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or embed_dim <= 0:
            raise ValueError("in_channels/embed_dim must be positive")
        if num_prototypes <= 1:
            raise ValueError("num_prototypes must be > 1")
        if not (1 <= min_topk <= num_prototypes):
            raise ValueError("min_topk must be in [1, num_prototypes]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 <= relation_momentum < 1.0):
            raise ValueError("relation_momentum must be in [0,1)")

        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.min_topk = int(min_topk)
        self.relation_weight = float(relation_weight)
        self.relation_warmup_epochs = int(relation_warmup_epochs)
        self.relation_ramp_epochs = int(relation_ramp_epochs)
        self.relation_momentum = float(relation_momentum)
        self.relation_smoothing = float(relation_smoothing)
        self.relation_log_floor = float(relation_log_floor)
        self.quant_scale = float(quant_scale)

        
        
        self.adapter = ConvNormAct(in_channels, embed_dim, kernel_size=1, norm=norm, act=True)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embed_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        
        
        self.output_norm = _make_norm_2d(embed_dim, norm)

        d = len(_DIRECTIONS)
        uniform = torch.full((d, num_prototypes, num_prototypes), 1.0 / num_prototypes)
        self.register_buffer("relation_prob", uniform)
        self.register_buffer(
            "relation_counts",
            torch.zeros(d, num_prototypes, num_prototypes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("relation_updates", torch.zeros((), dtype=torch.long))
        
        
        
        self.register_buffer("saved_epoch", torch.zeros((), dtype=torch.long))
        self._current_epoch = 0
        self._relation_alpha_cache = 0.0

    def _alpha_from_epoch(self, epoch: int) -> float:
        if epoch <= self.relation_warmup_epochs:
            return 0.0
        if self.relation_ramp_epochs <= 0:
            return 1.0
        return float(min(1.0, (epoch - self.relation_warmup_epochs) / float(self.relation_ramp_epochs)))

    @property
    def relation_alpha(self) -> float:
        return self._relation_alpha_cache

    def begin_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
        self._relation_alpha_cache = self._alpha_from_epoch(self._current_epoch)
        self.saved_epoch.fill_(self._current_epoch)
        self.relation_counts.zero_()

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        
        
        
        self._current_epoch = int(self.saved_epoch.detach().cpu().item())
        self._relation_alpha_cache = self._alpha_from_epoch(self._current_epoch)

    def relation_gate(self) -> torch.Tensor:
        p = self.relation_prob.float().clamp_min(_EPS)
        ent = _entropy(p, dim=-1)
        gate = 1.0 - ent / math.log(float(self.num_prototypes))
        return gate.clamp_(0.0, 1.0)

    @staticmethod
    def _direction_slices(h: int, w: int, dy: int, dx: int):
        if dy >= 0:
            cy0, cy1 = 0, h - dy
            ny0, ny1 = dy, h
        else:
            cy0, cy1 = -dy, h
            ny0, ny1 = 0, h + dy
        if dx >= 0:
            cx0, cx1 = 0, w - dx
            nx0, nx1 = dx, w
        else:
            cx0, cx1 = -dx, w
            nx0, nx1 = 0, w + dx
        return (slice(cy0, cy1), slice(cx0, cx1)), (slice(ny0, ny1), slice(nx0, nx1))

    def _relation_evidence(
        self,
        appearance_prob: torch.Tensor,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:












        b, n, k = appearance_prob.shape
        if n != h * w or k != self.num_prototypes:
            raise ValueError("appearance_prob shape does not match H/W/K")
        app = appearance_prob.view(b, h, w, k)
        rel = app.new_zeros((b, h, w, k))
        strength_num = app.new_zeros((b, h, w))
        consistency_num = app.new_zeros((b, h, w))
        reliability_den = app.new_zeros((b, h, w))
        valid_count = app.new_zeros((1, h, w))

        relation_prob = self.relation_prob.to(device=app.device, dtype=app.dtype)
        gates = self.relation_gate().to(device=app.device, dtype=app.dtype)

        for d, (dy, dx) in enumerate(_DIRECTIONS):
            center_slice, neigh_slice = self._direction_slices(h, w, dy, dx)
            cy, cx = center_slice
            ny, nx = neigh_slice
            center_app = app[:, cy, cx, :]       
            neigh_app = app[:, ny, nx, :]        

            trans = relation_prob[d]             
            gate = gates[d]                      
            row_max = trans.max(dim=-1).values.clamp_min(_EPS)

            
            
            compat = torch.matmul(neigh_app, trans.t())
            compat_norm = (compat / row_max.view(1, 1, 1, k)).clamp(0.0, 1.0)

            
            
            log_compat = compat_norm.clamp_min(math.exp(self.relation_log_floor)).log()
            corr = log_compat * gate.view(1, 1, 1, k)
            rel[:, cy, cx, :] += corr

            expected_gate = (center_app * gate.view(1, 1, 1, k)).sum(dim=-1)
            expected_gate_compat = (
                center_app * gate.view(1, 1, 1, k) * compat_norm
            ).sum(dim=-1)
            strength_num[:, cy, cx] += expected_gate
            consistency_num[:, cy, cx] += expected_gate_compat
            reliability_den[:, cy, cx] += expected_gate
            valid_count[:, cy, cx] += 1.0

        valid = valid_count.clamp_min(1.0)
        rel = rel / valid.unsqueeze(-1)
        structure_strength = strength_num / valid
        relation_consistency = torch.where(
            reliability_den > _EPS,
            consistency_num / reliability_den.clamp_min(_EPS),
            torch.ones_like(reliability_den),
        )
        return rel, structure_strength.clamp(0.0, 1.0), relation_consistency.clamp(0.0, 1.0)

    def _adaptive_sparse_projection(
        self,
        prob: torch.Tensor,
        suspicion: torch.Tensor,
        asp_alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        k = self.num_prototypes
        min_k = self.min_topk
        if asp_alpha <= 0.0 or min_k >= k:
            topk = torch.full_like(suspicion, k, dtype=torch.long)
            return prob, topk

        k_float = k - float(asp_alpha) * suspicion * float(k - min_k)
        topk = torch.round(k_float).to(torch.long).clamp_(min_k, k)

        order = _deterministic_order(prob, quant_scale=self.quant_scale)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(k, device=prob.device, dtype=order.dtype)
        rank_values = rank_values.view(*([1] * (order.dim() - 1)), k).expand_as(order)
        ranks.scatter_(-1, order, rank_values)
        mask = ranks < topk.unsqueeze(-1)

        sparse = prob * mask.to(dtype=prob.dtype)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        return sparse, topk

    def forward(self, x: torch.Tensor, return_aux: bool = True):
        if x.dim() != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                "Expected [B,{},H,W], got {}".format(self.in_channels, tuple(x.shape))
            )
        z_map = self.adapter(x)
        b, d, h, w = z_map.shape
        tokens = z_map.flatten(2).transpose(1, 2).contiguous()
        tokens = self.token_norm(tokens)
        tokens_n = F.normalize(tokens.float(), dim=-1, eps=_EPS).to(dtype=tokens.dtype)
        proto_n = F.normalize(self.prototypes.float(), dim=-1, eps=_EPS).to(dtype=tokens.dtype)

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
        out = normal_tokens.transpose(1, 2).contiguous().view(b, d, h, w)
        out = self.output_norm(out)

        if not return_aux:
            return out, {}

        usage = sparse_prob.mean(dim=(0, 1)).clamp_min(_EPS)
        usage = usage / usage.sum()
        proto_k_eff = _entropy(usage, dim=0).exp()
        token_entropy = _entropy(appearance_prob, dim=-1).mean()
        token_k_eff = token_entropy.exp()
        gate = self.relation_gate().to(device=x.device)
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
            "relation_gate_mean": gate.mean().detach(),
            "relation_gate_max": gate.max().detach(),
            "relation_alpha": x.new_tensor(float(alpha)).detach(),
        }
        return out, aux

    @torch.no_grad()
    def accumulate_relation_counts(self, appearance_prob: torch.Tensor, h: int, w: int) -> None:






        epoch = self._current_epoch
        if epoch < self.relation_warmup_epochs:
            return
        if appearance_prob.dim() != 3 or appearance_prob.size(1) != h * w:
            raise ValueError("appearance_prob must be [B,H*W,K]")

        order = _deterministic_order(appearance_prob, quant_scale=self.quant_scale)
        hard = order[..., 0].view(appearance_prob.size(0), h, w)
        k = self.num_prototypes

        for d, (dy, dx) in enumerate(_DIRECTIONS):
            center_slice, neigh_slice = self._direction_slices(h, w, dy, dx)
            cy, cx = center_slice
            ny, nx = neigh_slice
            center = hard[:, cy, cx].reshape(-1).to(torch.int64)
            neigh = hard[:, ny, nx].reshape(-1).to(torch.int64)
            flat = center * k + neigh
            counts = torch.bincount(flat, minlength=k * k).view(k, k)
            self.relation_counts[d].add_(counts)

    @torch.no_grad()
    def finalize_relation_epoch(self) -> bool:

        epoch = self._current_epoch
        if epoch < self.relation_warmup_epochs:
            self.relation_counts.zero_()
            return False
        counts = self.relation_counts.to(dtype=torch.float64)
        counts = counts + float(self.relation_smoothing)
        row_sum = counts.sum(dim=-1, keepdim=True)
        epoch_prob = counts / row_sum.clamp_min(_EPS)
        epoch_prob = epoch_prob.to(device=self.relation_prob.device, dtype=self.relation_prob.dtype)
        m = self.relation_momentum
        self.relation_prob.mul_(m).add_(epoch_prob, alpha=1.0 - m)
        self.relation_prob.div_(self.relation_prob.sum(dim=-1, keepdim=True).clamp_min(_EPS))
        self.relation_updates.add_(1)
        self.relation_counts.zero_()
        return True


