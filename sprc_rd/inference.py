













from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .checkpoint import load_model
from .state_residual_calibration import PrototypeStateResidualCalibrator
from .runtime import CUDAGraphResidualCalibrator, build_v4_drop_i32_cudagraph


class OptimizedAnomalyDetector:


    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        *,
        input_size: int = 256,
        outer_impl_override: str = "",
        sigma: float = -1.0,
        use_cuda_graph: bool = True,
        audit: bool = True,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if use_cuda_graph and self.device.type != "cuda":
            raise RuntimeError("CUDA Graph deployment requires a CUDA device")

        self.input_size = int(input_size)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.audit = bool(audit)

        self.model, self.checkpoint, self.model_config = load_model(
            checkpoint_path,
            self.device,
            outer_impl_override=outer_impl_override,
        )
        self.residual_calibrator = PrototypeStateResidualCalibrator.from_checkpoint(
            self.checkpoint, sigma=sigma
        ).to(self.device).eval()

        self.bottleneck_runner = None
        self.residual_calibrator_runner = None
        self.setup_audit: Dict[str, object] = {}
        self._input_signature = None

    def _validate_image(self, image: torch.Tensor):
        if image.device != self.device:
            raise ValueError("input tensor must already be on {}".format(self.device))
        if image.dtype != torch.float32:
            raise ValueError("finalized deployment expects FP32 input")
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("expected input shape [1,3,H,W]")
        if tuple(image.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                "expected spatial size {}x{}".format(self.input_size, self.input_size)
            )

    @torch.inference_mode()
    def _setup_cuda_graphs(self, example_image: torch.Tensor):
        teacher = [x.detach() for x in self.model.encoder(example_image)]
        self.bottleneck_runner, bottleneck_audit = build_v4_drop_i32_cudagraph(
            self.model.bottleneck,
            teacher,
            return_aux=True,
            audit=self.audit,
            capture_warmup=5,
        )
        self.model.set_bottleneck_runner(self.bottleneck_runner)

        teacher, student, aux = self.model.forward_features(
            example_image, return_aux=True
        )
        self.residual_calibrator_runner = CUDAGraphResidualCalibrator(
            self.residual_calibrator,
            teacher,
            student,
            aux,
            out_hw=(self.input_size, self.input_size),
            warmup=5,
            audit=self.audit,
        )
        self.setup_audit = {
            "bottleneck": bottleneck_audit,
            "residual_calibrator": self.residual_calibrator_runner.audit_report,
        }
        self._input_signature = (
            tuple(example_image.shape),
            example_image.dtype,
            example_image.device,
        )

    @torch.inference_mode()
    def forward_tensor(self, image: torch.Tensor) -> torch.Tensor:





        self._validate_image(image)

        if self.use_cuda_graph:
            signature = (tuple(image.shape), image.dtype, image.device)
            if self.residual_calibrator_runner is None:
                self._setup_cuda_graphs(image)
            elif signature != self._input_signature:
                raise ValueError("input signature changed after CUDA Graph capture")

            teacher, student, aux = self.model.forward_features(
                image, return_aux=True
            )
            return self.residual_calibrator_runner(teacher, student, aux)

        teacher, student, aux = self.model.forward_features(
            image, return_aux=True
        )
        return self.residual_calibrator(
            teacher,
            student,
            aux,
            out_hw=(self.input_size, self.input_size),
        )

    @torch.inference_mode()
    def forward_numpy(self, image: torch.Tensor) -> np.ndarray:

        amap = self.forward_tensor(image)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (
            amap[0, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )


__all__ = ["OptimizedAnomalyDetector"]
