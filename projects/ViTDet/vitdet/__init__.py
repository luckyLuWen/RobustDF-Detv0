# Copyright (c) OpenMMLab. All rights reserved.
"""Register the modules used by the released CNN-based configurations."""
from .dinov3 import DINOv3HubConvNeXtBackbone
from .hybrid_neck import DualPathNeckGlobalGate, DualPathNeckNoGate, GatedDualPathNeck
from .losses import RTMDetTupleFocalLoss

__all__ = [
    'DINOv3HubConvNeXtBackbone', 'GatedDualPathNeck',
    'DualPathNeckGlobalGate', 'DualPathNeckNoGate', 'RTMDetTupleFocalLoss',
]
