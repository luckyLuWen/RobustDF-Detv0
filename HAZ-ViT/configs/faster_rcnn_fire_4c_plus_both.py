"""
2-stage control group: baseline + DINOv3 ConvNeXt backbone + GatedDualPathNeck.
"""

_base_ = ['./faster_rcnn_fire_4c_plus_dino.py']

import inspect
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

model = dict(
    neck=dict(
        _delete_=True,
        type='GatedDualPathNeck',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4,
        start_level=0,
        num_csp_blocks=3,
        expand_ratio=0.5,
        init_fpn_weight=0.85,
        use_post_conv=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    rpn_head=dict(
        anchor_generator=dict(
            strides=[4, 8, 16, 32],
        ),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'backbone.model': dict(lr_mult=0.1),
            'backbone.model.norm': dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.model.norms': dict(lr_mult=0.1, decay_mult=0.0),
            'neck.gate_logits': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
)

work_dir = osp.join(_proj_root, 'work_dirs', 'haz_vit', 'configs', 'faster_rcnn_fire_4c_plus_both')
