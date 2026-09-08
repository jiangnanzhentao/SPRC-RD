import glob
import os
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_image_files(root: str, recursive: bool = False) -> List[str]:
    if not os.path.isdir(root):
        return []
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in glob.glob(os.path.join(root, pattern), recursive=recursive)
        if os.path.isfile(path) and path.lower().endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(files)


def discover_mvtec_classes(data_root: str, require_test: bool = True) -> List[str]:
    if not os.path.isdir(data_root):
        raise FileNotFoundError("data root not found: {}".format(data_root))
    names = []
    for name in sorted(os.listdir(data_root)):
        root = os.path.join(data_root, name)
        if not os.path.isdir(os.path.join(root, "train")):
            continue
        if require_test and not os.path.isdir(os.path.join(root, "test")):
            continue
        names.append(name)
    if not names:
        raise RuntimeError("no valid dataset classes under {}".format(data_root))
    return names


def get_data_transforms(size, input_size):
    data_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(input_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    target_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ]
    )
    return data_transform, target_transform


class MultiClassNormalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        class_names: Sequence[str],
        transform,
        normal_folder: str = "good",
        recursive: bool = True,
    ):
        self.class_names = list(class_names)
        self.transform = transform
        self.samples: List[Tuple[str, int, str]] = []
        for class_index, class_name in enumerate(self.class_names):
            train_root = os.path.join(data_root, class_name, "train")
            normal_root = os.path.join(train_root, normal_folder)
            source = normal_root if normal_folder and os.path.isdir(normal_root) else train_root
            paths = list_image_files(source, recursive=recursive)
            if not paths:
                raise RuntimeError("no normal training images for {}".format(class_name))
            self.samples.extend(
                (path, class_index, class_name) for path in paths
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, class_index, _ = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, class_index

    def class_counts(self):
        counts = {name: 0 for name in self.class_names}
        for _, _, name in self.samples:
            counts[name] += 1
        return counts
