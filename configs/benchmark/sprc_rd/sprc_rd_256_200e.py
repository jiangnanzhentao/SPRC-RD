from argparse import Namespace

import torchvision.transforms.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from configs.__base__ import cfg_common, cfg_dataset_default, cfg_model_sprc_rd


class cfg(cfg_common, cfg_dataset_default, cfg_model_sprc_rd):
    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_sprc_rd.__init__(self)
        self.metrics = [
            "mAUROC_sp_max",
            "mAP_sp_max",
            "mF1_max_sp_max",
            "mAUPRO_px",
            "mAUROC_px",
            "mAP_px",
            "mF1_max_px",
        ]
        self.evaluator.kwargs = dict(
            metrics=self.metrics,
            pooling_ks=None,
            max_step_aupro=100,
            use_adeval=True,
        )
        self.data.resize_shape = [self.size, self.size]
        self.data.train_transforms = [
            dict(type="Resize", size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type="CenterCrop", size=(self.size, self.size)),
            dict(type="ToTensor"),
            dict(
                type="Normalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                inplace=True,
            ),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(type="Resize", size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type="CenterCrop", size=(self.size, self.size)),
            dict(type="ToTensor"),
        ]
        self.sprc_rd = Namespace(
            state_residual_batch_size=128,
            state_residual_num_workers=4,
            state_residual_export_npz=False,
            pseudo_fraction=0.20,
            pseudo_rd_weight=0.50,
            mask_min_blocks=1,
            mask_max_blocks=2,
            mask_min_side=1,
            mask_max_side=5,
            mask_stripe_prob=0.5,
            mask_margin=1,
            mask_max_fraction=0.30,
            donor_min_shift=2,
            donor_max_shift=8,
            donor_candidates=4,
            max_source_overlap=0.10,
            pseudo_missing_probability=0.50,
            pseudo_missing_kernel=3,
            img_aug_start_epoch=80,
            img_aug_ramp_epochs=30,
            img_aug_fraction=0.15,
            img_aug_lambda_max=0.10,
            img_aug_min_fraction=0.05,
            img_aug_max_fraction=0.15,
            img_aug_masked_weight=1.2,
            img_aug_mean_fill_probability=0.25,
            img_aug_update_bn_stats=False,
            lambda_cf=0.20,
            lambda_gain=0.08,
            train_gain_margin=0.25,
            clean_gain_tolerance=0.05,
        )
