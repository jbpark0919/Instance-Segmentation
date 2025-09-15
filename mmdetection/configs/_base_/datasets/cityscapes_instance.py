# -*- coding: utf-8 -*-
# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/workspace/dataset/cityscapes'

# backend 설정
backend_args = None

# 훈련용 데이터 변환 파이프라인
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomResize', scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 테스트 데이터 변환 파이프라인
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 훈련 데이터 로더
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='/workspace/Jubin/mmdetection/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
            data_prefix=dict(img=data_root + '/leftImg8bit/train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args
        )
    )
)

# 검증 데이터 로더
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/workspace/Jubin/mmdetection/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img=data_root + '/leftImg8bit/val/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# 테스트 데이터 로더
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/workspace/Jubin/mmdetection/data/cityscapes/annotations/instancesonly_filtered_gtFine_test.json',
        data_prefix=dict(img=data_root + '/leftImg8bit/test/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

# 평가 설정
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file='/workspace/Jubin/mmdetection/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        metric=['bbox', 'segm'],
        backend_args=backend_args
    ),
    dict(
        type='CityScapesMetric',
        seg_prefix=data_root + '/gtFine/val',
        outfile_prefix='./work_dirs/cityscapes_metric/instance',
        backend_args=backend_args
    )
]

# 테스트 평가 설정 (Cityscapes 제출용 포맷)
test_evaluator = dict(
    type='CityScapesMetric',
    format_only=True,
    outfile_prefix='./work_dirs/cityscapes_metric/test'
)
