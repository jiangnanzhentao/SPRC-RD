from timm.models._registry import _model_entrypoints
from . import MODEL


for name in ("resnet18", "resnet34", "resnet50", "wide_resnet50_2"):
    MODEL.register_module(_model_entrypoints[name], "timm_{}".format(name))
