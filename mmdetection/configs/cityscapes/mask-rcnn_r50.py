# -*- coding: utf-8 -*-
_base_ = [
    '/workspace/Jubin/mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py',
    '/workspace/Jubin/mmdetection/configs/_base_/datasets/cityscapes_instance.py',
    '/workspace/Jubin/mmdetection/configs/_base_/schedules/schedule_1x.py', '/workspace/Jubin/mmdetection/configs/_base_/default_runtime.py'
]

# 데이터셋 설정
dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(2048, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/workspace/Jubin/mmdetection/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
        img_prefix='/workspace/dataset/cityscapes/leftImg8bit/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/workspace/Jubin/mmdetection/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix='/workspace/dataset/cityscapes/leftImg8bit/val/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/workspace/Jubin/mmdetection/data/cityscapes/annotations/instancesonly_filtered_gtFine_test.json',
        img_prefix='/workspace/dataset/cityscapes/leftImg8bit/test/',
        pipeline=train_pipeline)
)

# 모델 설정
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=8),  # Cityscapes에 맞게 클래스 개수 조정
        mask_head=dict(num_classes=8)
    )
)

# 학습 스케줄 설정
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='step', step=[8, 11])
total_epochs = 1

evaluation = dict(interval=1, metric=['bbox', 'segm'])
