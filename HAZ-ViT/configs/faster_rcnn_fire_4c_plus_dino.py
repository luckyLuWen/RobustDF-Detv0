"""
2-stage control group: baseline + DINOv3 ConvNeXt backbone.
"""

_base_ = ['./faster_rcnn_fire_4c_baseline.py']

import inspect
import os
import os.path as osp

def _find_proj_root(start_file):
    cur = osp.abspath(osp.dirname(start_file))
    for _ in range(8):
        if osp.isdir(osp.join(cur, 'HAZ-ViT')) and osp.isdir(osp.join(cur, 'datasets')):
            return cur
        parent = osp.abspath(osp.join(cur, '..'))
        if parent == cur:
            break
        cur = parent
    return osp.abspath(osp.join(osp.dirname(start_file), '..', '..'))

_proj_root = _find_proj_root(inspect.currentframe().f_code.co_filename)
dinov3_repo_dir = osp.join(_proj_root, 'HAZ-ViT', 'dinov3_repo')
_dinov3_weight_dir = osp.join(_proj_root, 'HAZ-ViT', 'weight_files')

dinov3_ckpt = None
if osp.isdir(_dinov3_weight_dir):
    _pths = sorted([f for f in os.listdir(_dinov3_weight_dir) if f.endswith('.pth')])
    _matches = [f for f in _pths if 'convnext_base' in f]
    if _matches:
        dinov3_ckpt = osp.join(_dinov3_weight_dir, _matches[0])

if dinov3_ckpt is None:
    raise FileNotFoundError(
        'Cannot find DINOv3 ConvNeXt-Base weights under: '
        f'{_dinov3_weight_dir}'
    )

model = dict(
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        _delete_=True,
        type='DINOv3HubConvNeXtBackbone',
        arch='dinov3_convnext_base',
        repo_dir=dinov3_repo_dir,
        weights=dinov3_ckpt,
        out_indices=(0, 1, 2, 3),
        apply_norm_on_outputs=True,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5,
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'backbone.model': dict(lr_mult=0.1),
            'backbone.model.norm': dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.model.norms': dict(lr_mult=0.1, decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
)

work_dir = osp.join(_proj_root, 'work_dirs', 'haz_vit', 'configs', 'faster_rcnn_fire_4c_plus_dino')
