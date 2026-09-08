# SPRC-RD

SPRC-RD is the cleaned ADer-based implementation of the proposed reverse-distillation method.

Method modules:

- Multi-scale Structural Prototype Projection (MSPP)
- Relation-Guided Prototype Intervention (RGPI)
- Prototype-State Residual Calibration (PSRC)

The default prototype counts are `(4, 4, 4)` and the default retained prototype counts are `(4, 2, 1)` for the fine, middle, and coarse scales.

## Data

Generate an ADer-style metadata file when necessary:

```bash
python generate_mvtec_meta.py --data-root /path/to/dataset
```

## Training

```bash
python run.py -c configs/benchmark/sprc_rd/sprc_rd_256_200e.py -m train data.root=/path/to/dataset
```

Training runs for 200 epochs. `training_state.pth` is overwritten after each epoch only for interruption recovery. At epoch 200, the code saves `sprc_rd_final.pth`, fits PSRC using normal training images, and embeds the calibration statistics into that same final model.

Resume an interrupted run:

```bash
python run.py -c configs/benchmark/sprc_rd/sprc_rd_256_200e.py -m train trainer.resume_dir=SPRC-RD_RUN_DIRECTORY data.root=/path/to/dataset
```

## Model and Training Log

The checkpoint used for the reported MVTec-AD results and its training log are provided below:

- Final model: [`checkpoints/sprc_rd_final.pth`](checkpoints/sprc_rd_final.pth)
- Training log: [`logs/mvtec_train.log`](logs/mvtec_train.log)

The checkpoint is obtained after 200 training epochs and contains the PSRC calibration statistics. 

## Testing

Evaluate the final checkpoint on MVTec-AD:

```bash
python run.py -c configs/benchmark/sprc_rd/sprc_rd_256_200e.py -m test data.root=/path/to/dataset model.kwargs.model_checkpoint_path=checkpoints/sprc_rd_final.pth
```

The checkpoint must contain the fitted PSRC calibration statistics. Testing an intermediate `training_state.pth` checkpoint is not supported.

## MVTec-AD Results

All results are reported in percent (%). The mean anomaly detection score (mAD) is the arithmetic mean of the seven image-level and pixel-level metrics.

| Category | I-AUROC | I-AP | I-F1max | AU-PRO | P-AUROC | P-AP | P-F1max |
|:--|--:|--:|--:|--:|--:|--:|--:|
| carpet | 98.596 | 99.604 | 97.143 | 96.600 | 99.338 | 74.773 | 68.390 |
| grid | 100.000 | 100.000 | 100.000 | 97.740 | 99.387 | 56.136 | 54.782 |
| leather | 100.000 | 100.000 | 100.000 | 99.135 | 99.723 | 69.290 | 64.380 |
| tile | 99.495 | 99.793 | 98.810 | 88.565 | 97.194 | 63.712 | 70.606 |
| wood | 99.211 | 99.745 | 98.361 | 94.424 | 97.089 | 60.625 | 61.091 |
| bottle | 100.000 | 100.000 | 100.000 | 96.880 | 98.992 | 84.253 | 77.979 |
| cable | 98.782 | 99.224 | 97.802 | 93.702 | 98.197 | 63.730 | 64.934 |
| capsule | 97.886 | 99.526 | 97.696 | 95.789 | 98.751 | 51.930 | 51.824 |
| hazelnut | 100.000 | 100.000 | 100.000 | 96.854 | 99.217 | 70.714 | 69.794 |
| metal_nut | 100.000 | 100.000 | 100.000 | 95.975 | 98.212 | 86.550 | 83.890 |
| pill | 97.736 | 99.606 | 97.143 | 96.779 | 97.996 | 77.612 | 72.554 |
| screw | 98.196 | 99.382 | 96.296 | 96.620 | 99.392 | 54.196 | 54.160 |
| toothbrush | 99.167 | 99.681 | 96.774 | 94.039 | 99.213 | 59.527 | 65.668 |
| transistor | 99.292 | 99.254 | 98.734 | 91.510 | 94.713 | 63.130 | 60.172 |
| zipper | 99.842 | 99.960 | 99.578 | 95.413 | 98.142 | 67.837 | 65.454 |
| **Average** | **99.213** | **99.718** | **98.556** | **95.335** | **98.370** | **66.934** | **65.712** |

**mAD: 89.120% (89.1%)**

## Image Inference

```bash
python tools/infer_anomaly_map.py --checkpoint runs/SPRC-RD_RUN_DIRECTORY/sprc_rd_final.pth --image /path/to/image.png --output anomaly_map.npz
```

The training loss and its weights retain the supplied implementation.
