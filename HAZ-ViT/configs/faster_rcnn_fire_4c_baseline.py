"""
2-stage control group: baseline (ResNet50 + FPN).
"""

_base_ = ['./faster_rcnn_fire_4c_2stage_common.py']

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

work_dir = osp.join(_proj_root, 'work_dirs', 'haz_vit', 'configs', 'faster_rcnn_fire_4c_baseline')
