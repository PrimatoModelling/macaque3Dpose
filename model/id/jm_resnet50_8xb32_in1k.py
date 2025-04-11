_base_ = [
    './resnet50.py',
    './imagenet_bs256.py', './default_runtime.py'
]

# ---- Model configs ----
model = dict(
    backbone=dict(
        init_cfg = dict(
            type='Pretrained', 
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth', 
            prefix='backbone')
    ),
    head=dict(
        num_classes=6, ###jm### データセットに合わせて変更
        topk = (1, )
    ))

# ---- Dataset configs ----
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/id_macaque/train',    ###jm### データセットに合わせて変更
        classes='data/id_macaque/classes.txt',   ###jm### データセットに合わせて変更
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/id_macaque/test',   ###jm### データセットに合わせて変更
        ann_file='data/id_macaque/test/annotations.txt',   ###jm### データセットに合わせて変更
        classes='data/id_macaque/classes.txt',   ###jm### データセットに合わせて変更
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/id_macaque/test',   ###jm### データセットに合わせて変更
        ann_file='data/id_macaque/test/annotations.txt',   ###jm### データセットに合わせて変更
        classes='data/id_macaque/classes.txt',   ###jm### データセットに合わせて変更
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )})


# ---- Schedule configs ----
# Usually in fine-tuning, we need a smaller learning rate and less training epochs.
# Specify the learning rate
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# Set the learning rate scheduler
#lr_config = dict(policy='step', step=1, gamma=0.1)
lr_config = dict(policy='step', step=[3, 6, 9])
runner = dict(type='EpochBasedRunner', max_epochs=10)

# ---- Runtime configs ----
# Output training log every 10 iterations.
log_config = dict(interval=10)
