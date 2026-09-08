
from __future__ import annotations

from typing import Sequence

import torch


class CUDAGraphBottleneck:






    def __init__(self, bottleneck, example_features: Sequence[torch.Tensor], return_aux: bool = False, warmup: int = 3):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph requires CUDA")
        if len(example_features) != 3 or any(x.device.type != "cuda" for x in example_features):
            raise ValueError("example_features must be three CUDA tensors")
        self.bottleneck = bottleneck.eval()
        self.return_aux = bool(return_aux)
        
        
        with torch.inference_mode(False), torch.no_grad():
            self.static_inputs = [torch.empty_like(x).copy_(x) for x in example_features]
        self._signature = [(tuple(x.shape), x.dtype, x.device) for x in self.static_inputs]

        side = torch.cuda.Stream(device=example_features[0].device)
        side.wait_stream(torch.cuda.current_stream(example_features[0].device))
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(max(1, int(warmup))):
                bottleneck(self.static_inputs, return_aux=self.return_aux)
        torch.cuda.current_stream(example_features[0].device).wait_stream(side)
        
        
        torch.cuda.synchronize(example_features[0].device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.static_output, self.static_aux = bottleneck(
                self.static_inputs, return_aux=self.return_aux
            )

    def _validate(self, features: Sequence[torch.Tensor]) -> None:
        if len(features) != 3:
            raise ValueError("SPRC-RD expects exactly three teacher features")
        actual = [(tuple(x.shape), x.dtype, x.device) for x in features]
        if actual != self._signature:
            raise ValueError(f"CUDA Graph shape/dtype/device mismatch: expected {self._signature}, got {actual}")

    def __call__(self, features: Sequence[torch.Tensor]):
        self._validate(features)
        for static, fresh in zip(self.static_inputs, features):
            static.copy_(fresh)
        self.graph.replay()
        return self.static_output, self.static_aux


def graph_audit(eager_output: torch.Tensor, graph_output: torch.Tensor):
    ref = eager_output.detach().float()
    got = graph_output.detach().float()
    diff = (ref - got).abs()
    cosine = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0)
    result = {
        "max_abs": float(diff.max().cpu()),
        "mean_abs": float(diff.mean().cpu()),
        "cosine": float(cosine.cpu()),
    }
    result["pass"] = (
        result["max_abs"] <= 1e-4
        and result["mean_abs"] <= 1e-5
        and result["cosine"] >= 0.999999
    )
    return result
