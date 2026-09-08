
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from PIL import Image

from sprc_rd.data import get_data_transforms
from sprc_rd.inference import OptimizedAnomalyDetector


def load_image(path: str, input_size: int, device: torch.device) -> torch.Tensor:
    transform, _ = get_data_transforms(input_size, input_size)
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device=device, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="output .npz path")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--outer_impl",
        choices=["auto", "ader"],
        default="auto",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=-1.0,
        help="negative value uses the calibrated/checkpoint sigma",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="disable CUDA Graph deployment and use eager inference",
    )
    parser.add_argument(
        "--skip_audit",
        action="store_true",
        help="skip one-time optimized-vs-reference setup audit",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    override = "" if args.outer_impl == "auto" else args.outer_impl

    detector = OptimizedAnomalyDetector(
        args.checkpoint,
        str(device),
        input_size=args.input_size,
        outer_impl_override=override,
        sigma=args.sigma,
        use_cuda_graph=not args.eager,
        audit=not args.skip_audit,
    )

    image = load_image(args.image, args.input_size, device)
    anomaly_map = detector.forward_numpy(image)
    image_score = float(np.max(anomaly_map))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        anomaly_map=anomaly_map.astype(np.float32, copy=False),
        image_score=np.asarray([image_score], dtype=np.float32),
    )

    print("saved:", output)
    print("anomaly_map:", anomaly_map.shape, anomaly_map.dtype)
    print("image_score:", image_score)
    print(
        "runtime:",
        "eager" if args.eager else "v4_drop_i32 CUDA Graph + prototype-state residual calibration CUDA Graph",
    )
    if detector.setup_audit:
        print("setup audit:", detector.setup_audit)


if __name__ == "__main__":
    main()
