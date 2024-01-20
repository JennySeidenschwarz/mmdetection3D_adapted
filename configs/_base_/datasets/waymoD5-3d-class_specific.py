dataset_type = 'AV2FeatherDataset' #'WaymoDataset'
root_dir = '/home/wiss/seidensc/Documents/project_clean_up/3DOpenWorldMOT/'
original_dataset_root = f'{root_dir}data/Waymo_Converted/'
flow_path = f'{root_dir}data/Waymo_Flow/'
filtered_file_path = f'SeMoLi/data_utils/Waymo_Converted_filtered/'
split_path = f'{root_dir}/SeMoLi/data_utils/new_seq_splits_Waymo_Converted_fixed_val/'
filter_static_evaluation = True
filter_static_training = True
class_agnostic = False
backend_args = None

class_names = ['TYPE_VECHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
metainfo = dict(classes=class_names)
in_channels = 5

point_cloud_range = [-50, -20, -2, 50, 20, 4] # [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)

train_pipeline = [
    dict(
        type='LoadPointsFromFileFeather',
        coord_type='LIDAR',
        load_dim=in_channels,
        use_dim=in_channels,
        backend_args=backend_args),
    dict(type='LoadAnnotations3DFeather', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3DFeather',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTransFeather',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilterFeather', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilterFeather', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputsFeather',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFileFeather',
        coord_type='LIDAR',
        load_dim=in_channels,
        use_dim=in_channels,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTransFeather',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3DFeather'),
            dict(
                type='PointsRangeFilterFeather', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputsFeather', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            label_path='',
            label_path2='',
            data_prefix=dict(
                pts=original_dataset_root, sweeps='training/velodyne'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            # load one frame every five frames
            load_interval=5,
            backend_args=backend_args,
            detection_type='train_detector',
            only_matched=False,
            filter_stat_before=filter_static_training,
            class_agnostic=class_agnostic,
            split_path=split_path)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            pts=original_dataset_root, sweeps='training/velodyne'),
        label_path='',
        label_path2='',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        detection_type='val_evaluation',
        class_agnostic=class_agnostic,
        filter_stat_before=filter_static_training,
        stat_as_ignore_region=False,
        split_path=split_path))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            pts=original_dataset_root, sweeps='training/velodyne'),
        label_path='',
        label_path2='',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        detection_type='val_detector',
        class_agnostic=class_agnostic,
        filter_stat_before=filter_static_training,
        stat_as_ignore_region=False,
        split_path=split_path))

val_evaluator = dict(
    type='AV2MetricFeather',
    data_root=f'{original_dataset_root}',
    backend_args=backend_args,
    convert_kitti_format=False,
    flow_path=flow_path,
    filtered_file_path=filtered_file_path,
    filter_static=filter_static_evaluation,
    root_dir=root_dir)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
