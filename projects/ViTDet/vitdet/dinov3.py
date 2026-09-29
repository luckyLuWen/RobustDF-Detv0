# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Sequence

import torch
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.registry import MODELS


def _first_tensor(seq: Sequence[Any]) -> Optional[Tensor]:
    for item in seq:
        if torch.is_tensor(item):
            return item
    return None


@MODELS.register_module()
class DINOv3HubBackbone(BaseModule):
    """A wrapper backbone that loads DINOv3 from PyTorch Hub and outputs (B, C, H, W).

    This is designed to plug DINOv3 ViT (typically patch=16) into ViTDet's
    SimpleFPN. The output feature map has stride ~= patch_size.

    Args:
        arch: Torch Hub entrypoint name, e.g. "dinov3_vitb16".
        repo_dir: Local clone of the DINOv3 repo. If provided, load with
            `source='local'` to avoid network usage.
        weights: Checkpoint URL or local path. For local-only usage, pass a
            local `.pth` file path.
        patch_size: Patch size used by the backbone (usually 16 for DINOv3).
        use_forward_features: Prefer calling `model.forward_features()` when
            available.
    """

    def __init__(
        self,
        arch: str = 'dinov3_vitb16',
        repo_dir: Optional[str] = None,
        weights: Optional[str] = None,
        patch_size: int = 16,
        use_forward_features: bool = True,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.arch = arch
        self.repo_dir = repo_dir
        self.weights = weights
        self.patch_size = patch_size
        self.use_forward_features = use_forward_features

        self.model = self._load_hub_model()

    def init_weights(self) -> None:
        """Keep hub-loaded weights and skip child re-initialization."""
        if self._is_init:
            return
        logger = MMLogger.get_current_instance()
        if logger is not None:
            logger.info(
                'Skip init_weights for DINOv3HubBackbone because torch.hub '
                'already constructed/loaded backbone weights.'
            )
        self._is_init = True

    def _load_hub_model(self) -> nn.Module:
        logger = MMLogger.get_current_instance()

        if sys.version_info < (3, 10):
            raise RuntimeError(
                'DINOv3 repo requires Python >= 3.10 (it uses PEP604 `X | None` type hints). '
                f'Current: {sys.version.split()[0]}.'
            )

        if not hasattr(torch, 'compiler') or not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError(
                'DINOv3 repo requires a newer PyTorch build (needs `torch.compiler` and float8 support). '
                f'Current torch={torch.__version__}. Recommended: torch>=2.1.'
            )

        kwargs: Dict[str, Any] = {}
        if self.weights:
            kwargs['weights'] = self.weights

        try:
            if self.repo_dir:
                logger.info(
                    f'Loading DINOv3 via torch.hub (local repo): arch={self.arch}, '
                    f'repo_dir={self.repo_dir}, weights={self.weights}'
                )
                model = torch.hub.load(
                    self.repo_dir,
                    self.arch,
                    source='local',
                    **kwargs,
                )
            else:
                logger.info(
                    f'Loading DINOv3 via torch.hub (remote): arch={self.arch}, '
                    f'weights={self.weights}'
                )
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.arch,
                    **kwargs,
                )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                'Failed to load DINOv3 from torch.hub. '
                'Recommended setup: `git clone https://github.com/facebookresearch/dinov3.git '
                'HAZ-ViT/dinov3_repo` and pass `repo_dir=.../HAZ-ViT/dinov3_repo`.\n'
                f'Original error: {e}'
            ) from e

        return model

    def _to_feature_map(self, out: Any, x: Tensor) -> Tensor:
        # Case 1: already a feature map
        if torch.is_tensor(out) and out.ndim == 4:
            return out

        tokens: Optional[Tensor] = None
        if isinstance(out, dict):
            for key in (
                'x_norm_patchtokens',
                'patchtokens',
                'patch_tokens',
                'x_patchtokens',
                'x_patch_tokens',
                'last_hidden_state',
            ):
                if key in out and torch.is_tensor(out[key]):
                    tokens = out[key]
                    break
        elif hasattr(out, 'last_hidden_state') and torch.is_tensor(out.last_hidden_state):
            tokens = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            tokens = _first_tensor(out)
        elif torch.is_tensor(out):
            tokens = out

        if tokens is None:
            raise TypeError(f'Unsupported DINOv3 output type: {type(out)}')

        if tokens.ndim == 4:
            return tokens
        if tokens.ndim != 3:
            raise ValueError(f'Expected tokens to be 3D (B, N, C), got shape: {tuple(tokens.shape)}')

        b, n, c = tokens.shape
        h, w = x.shape[-2:]
        ph = h // self.patch_size
        pw = w // self.patch_size
        patch_len = ph * pw
        if patch_len <= 0:
            raise ValueError(f'Invalid patch_len={patch_len} from input hw=({h},{w}) and patch={self.patch_size}')
        if patch_len > n:
            raise ValueError(
                f'Not enough tokens (N={n}) to reshape into patch grid (ph*pw={patch_len}). '
                'Make sure input sizes are divisible by patch_size.'
            )

        # Robust to extra tokens (cls/registers): take the last patch_len tokens.
        patch_tokens = tokens[:, -patch_len:, :]
        feat = patch_tokens.reshape(b, ph, pw, c).permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: Tensor) -> Tensor:
        if self.use_forward_features and hasattr(self.model, 'forward_features'):
            out = self.model.forward_features(x)
        else:
            out = self.model(x)
        return self._to_feature_map(out, x)


@MODELS.register_module()
class DINOv3HubConvNeXtBackbone(BaseModule):
    """A wrapper backbone that loads DINOv3 ConvNeXt from torch.hub and outputs multi-level features.

    This is designed for standard FPN-based detectors (ATSS / RetinaNet style).

    Expected strides for ConvNeXt stages:
      - stage0: 4
      - stage1: 8
      - stage2: 16
      - stage3: 32

    Args:
        arch: Torch Hub entrypoint name, e.g. "dinov3_convnext_base".
        repo_dir: Local clone of the DINOv3 repo (recommended).
        weights: Checkpoint URL or local path.
        out_indices: Stage indices to output.
        apply_norm_on_outputs: Whether to apply DINOv3 per-stage norms on
            returned feature maps. Recommended to keep True.
    """

    def __init__(
        self,
        arch: str = 'dinov3_convnext_base',
        repo_dir: Optional[str] = None,
        weights: Optional[str] = None,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        apply_norm_on_outputs: bool = True,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.arch = arch
        self.repo_dir = repo_dir
        self.weights = weights
        self.out_indices = tuple(int(i) for i in out_indices)
        self.apply_norm_on_outputs = apply_norm_on_outputs

        if not self.out_indices:
            raise ValueError('out_indices must be non-empty.')
        if min(self.out_indices) < 0 or max(self.out_indices) > 3:
            raise ValueError(f'out_indices must be within [0, 3], got: {self.out_indices}')

        self.model = self._load_hub_model()

    def init_weights(self) -> None:
        """Keep hub-loaded weights and skip child re-initialization."""
        if self._is_init:
            return
        logger = MMLogger.get_current_instance()
        if logger is not None:
            logger.info(
                'Skip init_weights for DINOv3HubConvNeXtBackbone because '
                'torch.hub already constructed/loaded backbone weights.'
            )
        self._is_init = True

    def _load_hub_model(self) -> nn.Module:
        logger = MMLogger.get_current_instance()

        if sys.version_info < (3, 10):
            raise RuntimeError(
                'DINOv3 repo requires Python >= 3.10 (it uses PEP604 `X | None` type hints). '
                f'Current: {sys.version.split()[0]}.'
            )

        if not hasattr(torch, 'compiler') or not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError(
                'DINOv3 repo requires a newer PyTorch build (needs `torch.compiler` and float8 support). '
                f'Current torch={torch.__version__}. Recommended: torch>=2.1.'
            )

        kwargs: Dict[str, Any] = {}
        if self.weights:
            kwargs['weights'] = self.weights

        try:
            if self.repo_dir:
                logger.info(
                    f'Loading DINOv3 ConvNeXt via torch.hub (local repo): arch={self.arch}, '
                    f'repo_dir={self.repo_dir}, weights={self.weights}'
                )
                model = torch.hub.load(
                    self.repo_dir,
                    self.arch,
                    source='local',
                    **kwargs,
                )
            else:
                logger.info(
                    f'Loading DINOv3 ConvNeXt via torch.hub (remote): arch={self.arch}, '
                    f'weights={self.weights}'
                )
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.arch,
                    **kwargs,
                )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                'Failed to load DINOv3 ConvNeXt from torch.hub. '
                'Recommended setup: `git clone https://github.com/facebookresearch/dinov3.git '
                'HAZ-ViT/dinov3_repo` and pass `repo_dir=.../HAZ-ViT/dinov3_repo`.\n'
                f'Original error: {e}'
            ) from e

        if not hasattr(model, 'downsample_layers') or not hasattr(model, 'stages'):
            raise TypeError(
                'Loaded DINOv3 model does not look like ConvNeXt. '
                f'Expected attributes `downsample_layers` and `stages`, got type: {type(model)}.'
            )

        return model

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        # Preferred path: use DINOv3's built-in intermediate feature API so
        # returned maps are consistent with official per-stage norm handling.
        if self.apply_norm_on_outputs and hasattr(self.model, 'get_intermediate_layers'):
            try:
                outs = self.model.get_intermediate_layers(  # type: ignore[attr-defined]
                    x,
                    n=list(self.out_indices),
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
                return tuple(outs)
            except Exception:  # noqa: BLE001
                # Fall back to manual stage traversal below.
                pass

        outs: list[Tensor] = []
        cur = x
        for i in range(4):
            cur = self.model.downsample_layers[i](cur)
            cur = self.model.stages[i](cur)
            if i in self.out_indices:
                outs.append(cur)
        return tuple(outs)
