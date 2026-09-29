"""
YOLOX-L ablation: replace neck only -> CSPNeXtPAFPN.
Backbone/head/loss/assigner remain YOLOX standard recipe.
"""

_base_ = ['./yolox_l_fire_4c_std.py']

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
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
)

work_dir = osp.join(_proj_root, 'work_dirs', 'haz_vit', 'configs', 'yolox_l_fire_4c_var_neck_cspnextpafpn')
