"""
Strict ablation on top of DINOv3-backbone RTMDet-L:
change optimizer only -> set backbone decay_mult=0.0.
"""

_base_ = ['./rtmdet_l_fire_4c_var_backbone_dinov3.py']

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

optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        custom_keys={
            'backbone': dict(decay_mult=0.0),
        },
    )
)

work_dir = osp.join(
    _proj_root, 'work_dirs', 'haz_vit', 'configs',
    'rtmdet_l_fire_4c_var_backbone_dinov3_opt_backbone_wd0')
