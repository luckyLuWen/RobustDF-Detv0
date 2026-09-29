"""
RTMDet-L ablation: replace cls loss only -> FocalLoss.
Backbone/neck/assigner remain RTMDet standard recipe.
"""

_base_ = ['./rtmdet_l_fire_4c_std.py']

custom_imports = dict(
    imports=['projects.ViTDet.vitdet'],
    allow_failed_imports=False,
)

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
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='RTMDetTupleFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='mean'
        ),
    ),
)

work_dir = osp.join(_proj_root, 'work_dirs', 'haz_vit', 'configs', 'rtmdet_l_fire_4c_var_loss_focal')
