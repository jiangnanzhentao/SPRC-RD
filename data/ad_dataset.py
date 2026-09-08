import json
import os
import random

import numpy as np
import torch.utils.data as data
from PIL import Image

from data import DATA
from util.data import get_img_loader


@DATA.register_module
class DefaultAD(data.Dataset):
    def __init__(self, cfg, train=True, transform=None, target_transform=None):
        self.root = cfg.data.root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = get_img_loader(cfg.data.loader_type)
        self.loader_target = get_img_loader(cfg.data.loader_type_target)
        with open(os.path.join(self.root, cfg.data.meta), "r", encoding="utf-8") as file:
            split = json.load(file)["train" if train else "test"]
        configured = cfg.data.cls_names
        if not isinstance(configured, list):
            configured = [configured]
        self.cls_names = list(split) if not configured else configured
        self.data_all = [sample for name in self.cls_names for sample in split[name]]
        if train:
            random.shuffle(self.data_all)
        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = self.data_all[index]
        image_path = os.path.join(self.root, sample["img_path"])
        image = self.loader(image_path)
        anomaly = int(sample["anomaly"])
        if anomaly:
            mask = np.asarray(
                self.loader_target(os.path.join(self.root, sample["mask_path"]))
            ) > 0
            mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        else:
            mask = Image.new("L", image.size, 0)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return {
            "img": image,
            "img_mask": mask,
            "cls_name": sample["cls_name"],
            "anomaly": anomaly,
            "img_path": image_path,
        }
