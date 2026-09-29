from typing import Any

from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.registry import MODELS


@MODELS.register_module()
class RTMDetTupleFocalLoss(FocalLoss):
    """FocalLoss wrapper for RTMDet targets.

    RTMDet cls head passes target as `(labels, assign_metrics)`.
    Standard FocalLoss expects only labels / one-hot tensor.
    """

    def forward(self,
                pred: Any,
                target: Any,
                weight: Any = None,
                avg_factor: Any = None,
                reduction_override: Any = None):
        if isinstance(target, (tuple, list)):
            target = target[0]
        return super().forward(
            pred=pred,
            target=target,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override)
