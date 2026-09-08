from typing import List

import torch


def rd_feature_loss(
    target_features: List[torch.Tensor], pred_features: List[torch.Tensor]
):
    if len(target_features) != len(pred_features):
        raise ValueError("teacher and student features must have the same length")
    loss = target_features[0].new_tensor(0.0)
    for target, prediction in zip(target_features, pred_features):
        if target.shape != prediction.shape:
            raise ValueError(
                "feature shape mismatch: {} vs {}".format(
                    tuple(target.shape), tuple(prediction.shape)
                )
            )
        loss = loss + torch.mean(
            1.0
            - torch.nn.functional.cosine_similarity(
                target.reshape(target.shape[0], -1),
                prediction.reshape(prediction.shape[0], -1),
                dim=1,
            )
        )
    return loss
