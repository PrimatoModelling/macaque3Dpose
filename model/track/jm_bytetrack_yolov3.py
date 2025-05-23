_base_ = [
    './mot_challenge.py', './default_runtime.py'
]

img_scale = (832, 832)
samples_per_gpu = 4

model = dict(
    type='ByteTrack',
    detector=dict(
        type='YOLOV3',
        backbone=dict(
            type='Darknet',
            depth=53,
            out_indices=(3, 4, 5),
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
        neck=dict(
            type='YOLOV3Neck',
            num_scales=3,
            in_channels=[1024, 512, 256],
            out_channels=[512, 256, 128]),
        bbox_head=dict(
            type='YOLOV3Head',
            num_classes=1,
            in_channels=[512, 256, 128],
            out_channels=[1024, 512, 256],
            anchor_generator=dict(
                type='YOLOAnchorGenerator',
                base_sizes=[[(116, 90), (156, 198), (373, 326)],
                            [(30, 61), (62, 45), (59, 119)],
                            [(10, 13), (16, 30), (33, 23)]],
                strides=[32, 16, 8]),
            bbox_coder=dict(type='YOLOBBoxCoder'),
            featmap_strides=[32, 16, 8],
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0,
                reduction='sum'),
            loss_conf=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0,
                reduction='sum'),
            loss_xy=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=2.0,
                reduction='sum'),
            loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type='GridAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0)),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            conf_thr=0.005,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100),

        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'weight/detection.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=5))

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(416, 416), (832, 832)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(832, 832),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = 'data/macaquegcage_coco2/'

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        _delete_=True,
        type='CocoDataset',
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline,
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
    test=dict(
        pipeline=test_pipeline,
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.001 / 8 * samples_per_gpu,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)

# some hyper parameters
total_epochs = 80
num_last_epochs = 10
resume_from = None
interval = 5

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))
