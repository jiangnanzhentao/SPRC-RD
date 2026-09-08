from PIL import Image


def pil_loader(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def pil_loader_l(path):
    with Image.open(path) as image:
        return image.convert("L")


def get_img_loader(loader_type):
    loaders = {"pil": pil_loader, "pil_L": pil_loader_l}
    if loader_type not in loaders:
        raise ValueError("unsupported image loader: {}".format(loader_type))
    return loaders[loader_type]
