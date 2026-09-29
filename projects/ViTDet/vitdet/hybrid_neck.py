from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList
from torch import Tensor, nn

from mmdet.models.necks import CSPNeXtPAFPN, FPN
from mmdet.registry import MODELS


def _build_post_convs(
    num_outs: int,
    out_channels: int,
    norm_cfg: Optional[dict],
    act_cfg: Optional[dict],
    use_post_conv: bool,
) -> Optional[ModuleList]:
    if not use_post_conv:
        return None
    return ModuleList([
        ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        for _ in range(num_outs)
    ])


@MODELS.register_module()
class GatedDualPathNeck(BaseModule):
    """Fuse FPN and CSPNeXtPAFPN with learnable per-level gates."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 256,
        num_outs: int = 3,
        start_level: int = 0,
        num_csp_blocks: int = 3,
        expand_ratio: float = 0.5,
        init_fpn_weight: float = 0.8,
        use_post_conv: bool = True,
        norm_cfg: Optional[dict] = dict(type='BN'),
        act_cfg: Optional[dict] = dict(type='SiLU', inplace=True),
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if not (0.01 < init_fpn_weight < 0.99):
            raise ValueError('init_fpn_weight must be in (0.01, 0.99).')

        self.num_outs = num_outs
        self.use_post_conv = use_post_conv

        self.fpn = FPN(
            in_channels=list(in_channels),
            out_channels=out_channels,
            start_level=start_level,
            num_outs=num_outs,
        )
        self.pafpn = CSPNeXtPAFPN(
            in_channels=list(in_channels),
            out_channels=out_channels,
            num_csp_blocks=num_csp_blocks,
            expand_ratio=expand_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        init_logit = math.log(init_fpn_weight / (1.0 - init_fpn_weight))
        self.gate_logits = nn.Parameter(torch.full((num_outs,), init_logit))

        self.post_convs = _build_post_convs(
            num_outs=num_outs,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            use_post_conv=use_post_conv)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        fpn_feats = self.fpn(inputs)
        pafpn_feats = self.pafpn(inputs)
        if len(fpn_feats) != self.num_outs or len(pafpn_feats) != self.num_outs:
            raise RuntimeError('Unexpected neck output length mismatch.')

        gates = torch.sigmoid(self.gate_logits)
        outs = []
        for level in range(self.num_outs):
            fused = gates[level] * fpn_feats[level] + (1.0 - gates[level]) * pafpn_feats[level]
            if self.post_convs is not None:
                fused = self.post_convs[level](fused)
            outs.append(fused)
        return tuple(outs)


@MODELS.register_module()
class DualPathNeckGlobalGate(BaseModule):
    """Fuse FPN and CSPNeXtPAFPN with one shared learnable gate."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 256,
        num_outs: int = 3,
        start_level: int = 0,
        num_csp_blocks: int = 3,
        expand_ratio: float = 0.5,
        init_fpn_weight: float = 0.8,
        use_post_conv: bool = True,
        norm_cfg: Optional[dict] = dict(type='BN'),
        act_cfg: Optional[dict] = dict(type='SiLU', inplace=True),
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if not (0.01 < init_fpn_weight < 0.99):
            raise ValueError('init_fpn_weight must be in (0.01, 0.99).')

        self.num_outs = num_outs
        self.fpn = FPN(
            in_channels=list(in_channels),
            out_channels=out_channels,
            start_level=start_level,
            num_outs=num_outs,
        )
        self.pafpn = CSPNeXtPAFPN(
            in_channels=list(in_channels),
            out_channels=out_channels,
            num_csp_blocks=num_csp_blocks,
            expand_ratio=expand_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        init_logit = math.log(init_fpn_weight / (1.0 - init_fpn_weight))
        self.gate_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.post_convs = _build_post_convs(
            num_outs=num_outs,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            use_post_conv=use_post_conv)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        fpn_feats = self.fpn(inputs)
        pafpn_feats = self.pafpn(inputs)
        if len(fpn_feats) != self.num_outs or len(pafpn_feats) != self.num_outs:
            raise RuntimeError('Unexpected neck output length mismatch.')

        gate = torch.sigmoid(self.gate_logit)
        outs = []
        for level in range(self.num_outs):
            fused = gate * fpn_feats[level] + (1.0 - gate) * pafpn_feats[level]
            if self.post_convs is not None:
                fused = self.post_convs[level](fused)
            outs.append(fused)
        return tuple(outs)


@MODELS.register_module()
class DualPathNeckNoGate(BaseModule):
    """Fuse FPN and CSPNeXtPAFPN with fixed ratio (no learnable gate)."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 256,
        num_outs: int = 3,
        start_level: int = 0,
        num_csp_blocks: int = 3,
        expand_ratio: float = 0.5,
        fpn_weight: float = 0.5,
        use_post_conv: bool = True,
        norm_cfg: Optional[dict] = dict(type='BN'),
        act_cfg: Optional[dict] = dict(type='SiLU', inplace=True),
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if not (0.0 <= fpn_weight <= 1.0):
            raise ValueError('fpn_weight must be in [0, 1].')

        self.num_outs = num_outs
        self.fpn_weight = float(fpn_weight)
        self.fpn = FPN(
            in_channels=list(in_channels),
            out_channels=out_channels,
            start_level=start_level,
            num_outs=num_outs,
        )
        self.pafpn = CSPNeXtPAFPN(
            in_channels=list(in_channels),
            out_channels=out_channels,
            num_csp_blocks=num_csp_blocks,
            expand_ratio=expand_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.post_convs = _build_post_convs(
            num_outs=num_outs,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            use_post_conv=use_post_conv)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        fpn_feats = self.fpn(inputs)
        pafpn_feats = self.pafpn(inputs)
        if len(fpn_feats) != self.num_outs or len(pafpn_feats) != self.num_outs:
            raise RuntimeError('Unexpected neck output length mismatch.')

        gate = self.fpn_weight
        outs = []
        for level in range(self.num_outs):
            fused = gate * fpn_feats[level] + (1.0 - gate) * pafpn_feats[level]
            if self.post_convs is not None:
                fused = self.post_convs[level](fused)
            outs.append(fused)
        return tuple(outs)
