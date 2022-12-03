# model settings
model = dict(
    type='RotatedRepPoints',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained= True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5),
    bbox_head=dict(
        type='CFAHead',
        num_classes=16,
        in_channels=256,
        feat_channels=256,
        stacked_convs=3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        num_points=9,
        gradient_mul=0.3,
        transform_method='rotrect',
        topk=6,
        anti_factor=0.75,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
        test_cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_thr=0.4),
            max_per_img=2000),
            
        train_cfg=dict(
            init=dict(
                assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            refine=dict(
                assigner=dict(
                    type='MaxConvexIoUAssigner',
                    pos_iou_thr=0.1,
                    neg_iou_thr=0.1,
                    min_pos_iou=0,
                    ignore_iof_thr=-1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
    ))
dataset = dict(
    train=dict(
        type="DOTADataset",
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(type='RotatedRandomFlip', prob=0.5, direction="horizontal"),
            dict(type='RotatedRandomFlip', prob=0.5, direction="vertical"),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)
            
        ],
        batch_size=2,
        num_workers=4,
        shuffle=True,
        filter_empty_gt=False
    ),
    val=dict(
        type="DOTADataset",
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=2,
        num_workers=4,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        images_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,),
        ],
        num_workers=4,
        batch_size=1,
    )
)

optimizer = dict(
    type='SGD', 
    lr=0.01/4., #0.0,#0.01*(1/8.), 
    momentum=0.9, 
    weight_decay=0.0001,
    grad_clip=dict(
        max_norm=35, 
        norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[7, 10])


logger = dict(
    type="RunLogger")

max_epoch = 12
eval_interval = 1
checkpoint_interval = 1
log_interval = 50
