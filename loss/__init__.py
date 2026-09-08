from util.registry import Registry


LOSS = Registry("Loss")


def get_loss_terms(loss_terms, device="cpu"):
    terms = {}
    for config in loss_terms:
        kwargs = dict(config)
        loss_type = kwargs.pop("type")
        name = kwargs.pop("name")
        terms[name] = LOSS.get_module(loss_type)(**kwargs).to(device).eval()
    return terms


from .base_loss import CosLoss
