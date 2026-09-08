import copy

import torch

from util.registry import Registry


MODEL = Registry("Model")


def get_model(model_cfg):
    kwargs = copy.deepcopy(model_cfg.kwargs)
    pretrained = kwargs.pop("pretrained", False)
    checkpoint_path = kwargs.pop("checkpoint_path", "")
    strict = kwargs.pop("strict", True)
    model = MODEL.get_module(model_cfg.name)(pretrained=pretrained, **kwargs)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        model.load_state_dict(state, strict=strict)
    return model


from . import ad_factory, rd, sprc_rd
