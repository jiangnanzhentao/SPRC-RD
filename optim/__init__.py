from torch.optim import Adam


def get_optim(optim_kwargs, net, lr):
    kwargs = dict(optim_kwargs)
    name = kwargs.pop("name").lower()
    if name != "adam":
        raise ValueError("SPRC-RD uses the Adam optimizer")
    params = (
        net.get_optim_param_groups(lr)
        if hasattr(net, "get_optim_param_groups")
        else net.parameters()
    )
    return Adam(params=params, lr=lr, **kwargs)
