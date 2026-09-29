"""
YOLOX-L ablation: replace bbox loss only -> GIoULoss.
Backbone/neck/assigner remain YOLOX standard recipe.
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
    bbox_head=dict(
        loss_bbox=dict(
            _delete_=True,
            type='GIoULoss',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0,
        ),
    ),
)

work_dir = osp.join(_proj_root, 'work_dirs', 'haz_vit', 'configs', 'yolox_l_fire_4c_var_loss_giou')
