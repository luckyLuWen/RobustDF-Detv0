"""
Common training recipe for 2-stage four-way ablation on fire2/coco_4c.
"""

_base_ = [
    '../mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../mmdetection/configs/_base_/default_runtime.py',
]

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
data_root = osp.join(_proj_root, 'datasets', 'fire2', 'coco_4c') + osp.sep

dataset_type = 'CocoDataset'
backend_args = None
image_scale = (896, 896)

metainfo = dict(
    classes=('car_fire', 'car_nofire', 'lkyw_fire', 'lkyw_nofire'),
    palette=[(220, 20, 60), (0, 255, 0), (255, 165, 0), (0, 191, 255)],
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    backend_args=backend_args,
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    backend_args=backend_args,
)

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=osp.join(
                _proj_root,
                'pretrained',
                'torchvision',
                'resnet50-0676ba61.pth',
            ),
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=4,
            reg_class_agnostic=True,
        ),
    ),
)

max_epochs = 70

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.0)}),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=max_epochs, eta_min=1e-6),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='coco/bbox_mAP', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=20),
)

randomness = dict(seed=0)
