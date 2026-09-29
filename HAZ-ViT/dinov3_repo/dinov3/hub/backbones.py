# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import os

from enum import Enum

from typing import List, Optional, Union

from urllib.parse import urlparse

from pathlib import Path

import torch

from .utils import DINOV3_BASE_URL

class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"

def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")

def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()

def _make_dinov3_convnext_model_url(
    *,
    compact_arch_name: str = "convnext_base",
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""

    model_dir = f"{model_name}_{compact_arch_name}"
    model_filename = f"{model_name}_{compact_arch_name}_pretrain_{weights_name}{hash_suffix}.pth"
    return os.path.join(DINOV3_BASE_URL, model_dir, model_filename)

def _make_dinov3_convnext(
    in_chans: int = 3,
    depths: List[int] = [3, 3, 27, 3],
    dims: List[int] = [128, 256, 512, 1024],
    compact_arch_name: str = "convnext_base",
    drop_path_rate: float = 0.0,
    layer_scale_init_value: float = 1e-6,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    **kwargs,
):
    from ..models.convnext import ConvNeXt

    model_kwargs = dict(
        in_chans=in_chans,
        depths=depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )
    model_kwargs.update(**kwargs)
    model = ConvNeXt(**model_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_convnext_model_url(
                compact_arch_name=compact_arch_name,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model

def dinov3_convnext_base(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "801f2ba9"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["base"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_base",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model
