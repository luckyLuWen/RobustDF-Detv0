"""
YOLOX-L innovation trial: DINOv3 backbone + GatedDualPathNeck.
"""

_base_ = ['./yolox_l_fire_4c_var_backbone_dinov3.py']

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
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_outs=3,
        start_level=0,
        num_csp_blocks=3,
        expand_ratio=0.5,
        init_fpn_weight=0.85,
        use_post_conv=True,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.,
        bias_decay_mult=0.,
        bypass_duplicate=True,
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=0.0),
            'neck.gate_logits': dict(decay_mult=0.0),
        },
    ),
)

work_dir = osp.join(
    _proj_root, 'work_dirs', 'haz_vit', 'configs',
    'yolox_l_fire_4c_innov_gated_dualneck_dinov3')
