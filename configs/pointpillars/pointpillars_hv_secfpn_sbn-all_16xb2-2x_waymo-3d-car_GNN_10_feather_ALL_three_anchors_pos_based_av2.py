_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_waymo_PC_REST_av2.py',
    '../_base_/datasets/av2-3d-car_GNN_10_feather_pos_based.py',
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
            ranges=[[-74.88, -74.88, 0, 74.88, 74.88, 0],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0],],
            sizes=[
                [0.75, 0.75, 0.75],  # car
                [1.5, 0.75, 1.5],  # pedestrian
                [4.5, 2, 1.5],  # cyclist
                [6, 2.5, 2],
                [9, 3, 3],
                [13, 3, 3.5],
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
# vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
