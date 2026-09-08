from . import TRANSFORMS


def get_transforms(cfg, train, cfg_transforms):
    del cfg, train
    transform_list = []
    for transform_config in cfg_transforms:
        kwargs = dict(transform_config)
        transform_type = kwargs.pop("type")
        transform = TRANSFORMS.get_module(transform_type)(**kwargs)
        transform_list.extend(transform if isinstance(transform, list) else [transform])
    return TRANSFORMS.get_module("Compose")(transform_list)
