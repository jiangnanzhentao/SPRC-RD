from torchvision import transforms

from . import TRANSFORMS


for name in ("Compose", "Resize", "CenterCrop", "ToTensor", "Normalize"):
    TRANSFORMS.register_module(getattr(transforms, name), name=name)
