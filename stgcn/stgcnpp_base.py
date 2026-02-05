_base_ = 'default_runtime.py'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco_wholebody', mode='spatial'),
        in_channels=6),
    cls_head=dict(type='GCNHead', num_classes=2, in_channels=256, dropout=0.5))

dataset_type = 'PoseDataset'
ann_file = 'k_fold/data/skeleton/bcm_master_annotation_fold1.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco_wholebody', feats=['jm', 'bm']),
    dict(type='UniformSampleFrames', clip_len=90),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco_wholebody', feats=['jm', 'bm']),
    dict(
        type='UniformSampleFrames', clip_len=90, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco_wholebody', feats=['jm', 'bm']),
    dict(
        type='UniformSampleFrames', clip_len=90, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train')))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='xsub_test',
        test_mode=True))

val_evaluator = [dict(type='AccMetric'), dict(type='LossMetric', prefix='val')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# param_scheduler = [
#     dict(
#         type='CosineAnnealingLR',
#         eta_min=0,
#         T_max=16,
#         by_epoch=True,
#         convert_to_iter_based=True)
# ]
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=1e-06, type='AdamW', weight_decay=0.001),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))

# default_hooks = dict(
#     checkpoint=dict(
#         interval=1, 
#         save_best='auto',  # Save the best model based on validation metrics
#         rule='greater'     # Higher accuracy is better
#     ), 
#     logger=dict(interval=100)
# )

default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
