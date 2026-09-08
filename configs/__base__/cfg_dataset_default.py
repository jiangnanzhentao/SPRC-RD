from argparse import Namespace

import torchvision.transforms.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class cfg_dataset_default(Namespace):
    def __init__(self):
        super().__init__()
        self.data = Namespace()
        self.data.sampler = "naive"
        self.data.loader_type = "pil"
        self.data.loader_type_target = "pil_L"
        self.data.type = "DefaultAD"
        self.data.root = "data/mvtec"
        self.data.meta = "meta.json"
        self.data.cls_names = []
        self.data.train_transforms = [
            dict(type="Resize", size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
            dict(type="CenterCrop", size=(256, 256)),
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
            dict(type="Resize", size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
            dict(type="CenterCrop", size=(256, 256)),
            dict(type="ToTensor"),
        ]
