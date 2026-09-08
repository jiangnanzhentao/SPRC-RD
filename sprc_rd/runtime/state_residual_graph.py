
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

_AUX_SCALES = ("fine", "mid", "coarse")
_AUX_KEYS = ("cf_q", "cf_confidence")
_TOP_AUX_KEYS = ("prototype_state_confidence",)


def _tensor_audit(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    ref = reference.detach().float()
    got = candidate.detach().float()
    diff = (ref - got).abs()
    cosine = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0)
    result = {
        "max_abs": float(diff.max().cpu()),
        "mean_abs": float(diff.mean().cpu()),
        "cosine": float(cosine.cpu()),
    }
    result["pass"] = bool(
        result["max_abs"] <= 1e-4
        and result["mean_abs"] <= 1e-5
        and result["cosine"] >= 0.999999
    )
    return result


def _clone_tensor(x: torch.Tensor) -> torch.Tensor:
    
    
    with torch.inference_mode(False), torch.no_grad():
        return torch.empty_like(x).copy_(x)


def _clone_aux(aux):
    result = {}
    for scale in _AUX_SCALES:
        result[scale] = {
            key: _clone_tensor(aux[scale][key])
            for key in _AUX_KEYS
        }
    for key in _TOP_AUX_KEYS:
        result[key] = _clone_tensor(aux[key])
    return result


class CUDAGraphResidualCalibrator:


    def __init__(
        self,
        runtime,
        teacher_features: Sequence[torch.Tensor],
        student_features: Sequence[torch.Tensor],
        aux,
        *,
        out_hw: Tuple[int, int],
        warmup: int = 5,
        audit: bool = True,
    ):
        if len(teacher_features) != 3 or len(student_features) != 3:
            raise ValueError("prototype-state residual calibration expects three teacher/student feature tensors")
        if any(x.device.type != "cuda" for x in teacher_features):
            raise RuntimeError("prototype-state residual calibration CUDA Graph requires CUDA tensors")

        self.runtime = runtime.eval()
        self.out_hw = tuple(int(v) for v in out_hw)
        self.static_teacher = [_clone_tensor(x) for x in teacher_features]
        self.static_student = [_clone_tensor(x) for x in student_features]
        self.static_aux = _clone_aux(aux)
        self._teacher_signature = [(tuple(x.shape), x.dtype, x.device) for x in self.static_teacher]
        self._student_signature = [(tuple(x.shape), x.dtype, x.device) for x in self.static_student]

        eager_reference = None
        if audit:
            with torch.inference_mode():
                eager_reference = self.runtime(
                    teacher_features, student_features, aux, out_hw=self.out_hw
                )
                torch.cuda.synchronize(teacher_features[0].device)

        side = torch.cuda.Stream(device=teacher_features[0].device)
        side.wait_stream(torch.cuda.current_stream(teacher_features[0].device))
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(max(1, int(warmup))):
                self.runtime(
                    self.static_teacher,
                    self.static_student,
                    self.static_aux,
                    out_hw=self.out_hw,
                )
        torch.cuda.current_stream(teacher_features[0].device).wait_stream(side)
        torch.cuda.synchronize(teacher_features[0].device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.static_output = self.runtime(
                self.static_teacher,
                self.static_student,
                self.static_aux,
                out_hw=self.out_hw,
            )

        self.audit_report = {}
        if audit:
            self.copy_inputs_(teacher_features, student_features, aux)
            self.graph.replay()
            torch.cuda.synchronize(teacher_features[0].device)
            self.audit_report = _tensor_audit(eager_reference, self.static_output)
            if not self.audit_report["pass"]:
                raise RuntimeError(
                    "prototype-state residual calibration CUDA Graph audit failed: {}".format(self.audit_report)
                )

    def _validate(self, teacher_features, student_features):
        teacher_signature = [(tuple(x.shape), x.dtype, x.device) for x in teacher_features]
        student_signature = [(tuple(x.shape), x.dtype, x.device) for x in student_features]
        if teacher_signature != self._teacher_signature:
            raise ValueError("teacher feature shape/dtype/device changed after CUDA Graph capture")
        if student_signature != self._student_signature:
            raise ValueError("student feature shape/dtype/device changed after CUDA Graph capture")

    def copy_inputs_(self, teacher_features, student_features, aux):
        self._validate(teacher_features, student_features)
        for static, fresh in zip(self.static_teacher, teacher_features):
            static.copy_(fresh)
        for static, fresh in zip(self.static_student, student_features):
            static.copy_(fresh)
        for scale in _AUX_SCALES:
            for key in _AUX_KEYS:
                self.static_aux[scale][key].copy_(aux[scale][key])
        for key in _TOP_AUX_KEYS:
            self.static_aux[key].copy_(aux[key])

    def __call__(self, teacher_features, student_features, aux):
        self.copy_inputs_(teacher_features, student_features, aux)
        self.graph.replay()
        return self.static_output


__all__ = ["CUDAGraphResidualCalibrator"]
