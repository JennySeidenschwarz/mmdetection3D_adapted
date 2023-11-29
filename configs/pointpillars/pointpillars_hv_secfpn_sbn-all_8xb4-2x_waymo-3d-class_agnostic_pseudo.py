_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo_PC_REST.py',
    '../_base_/datasets/waymoD5-3d-class_agnostic_pseudo.py',
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py',
]

# data settings
train_dataloader = dict(dataset=dict(dataset=dict(load_interval=1)))

# model settings
model = dict(
    type='MVXFasterRCNN',
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            # ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345]],
            ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0],
                    [-74.88, -74.88, -0.1188, 74.88, 74.88, -0.1188]],
            sizes=[
                [4.73, 2.08, 1.77],  # car
                [0.91, 0.84, 1.74],  # pedestrian
                [1.81, 0.84, 1.77]  # cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        pts=dict(
            assigner=dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
