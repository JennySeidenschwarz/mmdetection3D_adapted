# dataset settings
# D5 in the config name means the whole dataset is divided into 5 folds
# We only use one fold for efficient experiments
dataset_type = 'AV2FeatherDataset' #'WaymoDataset'

data_root = '/workspace/waymo_kitti_format/kitti_format/'
data_root_annotatons = f'/workspace/waymo_kitti_format_annotaions/kitti_format/'

data_root_annotatons_dets = '/workspace/ExchangeWorkspace/detections_train_detector/'
detection_name = 'GNN_motion_patterns_MORE_OR_LESS_FINAL_GNN_HIHI_0.1_0.1_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_3.162277660168379e-06_0.0031622776601683794_16000_16000__NS_MG_32_LN___P___MMMDPTT___PT_/train_detector/annotations_IoU3D.feather'

rel_annotations_dir = '../../waymo_kitti_format_annotaions/kitti_format'

original_dataset_root = f'/workspace/Argoverse2/'
backend_args = None

class_names = ['Car']
metainfo = dict(classes=class_names)

point_cloud_range = [-50, -20, -2, 50, 20, 4] # [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root_annotatons_dets,
    info_path=data_root_annotatons_dets + f'debug_training',
    data_root2=data_root_annotatons_dets,
    info_path2=data_root_annotatons_dets + f'debug_training/waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFileFeather',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
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
        load_dim=6,
        use_dim=5,
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
            data_root=data_root_annotatons_dets,
            ann_file='/workspace/mmdetection3d/waymo_debug_infos_train_dict.pkl', #'/workspace/mmdetection3d/data/pseudo_labels/kitti_format/RECOMPUTE_REMOVE_BUG_HEADING_90_EVAL_TRAIN_DET_0.9_0.9_all_egocomp_margin0.6_width25_nooracle_64_64_64_64_0.5_3.5_0.5_4_3.162277660168379e-06_0.0031622776601683794_16000_16000__NS_MG_32_2.0_LN__0/waymo_infos_train.pkl', # f'{data_root_annotatons}waymo_infos_train.pkl',
            pseudo_labels=f'{data_root_annotatons_dets}{detection_name}',
            data_prefix=dict(
                pts=f'{data_root}training/velodyne', sweeps='training/velodyne'),
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
            load_dir='/workspace/Argoverse2/train')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        # ann_file=f'{rel_annotations_dir}/waymo_infos_val.pkl',
        ann_file=f'{rel_annotations_dir}/waymo_infos_train.pkl',
        pseudo_labels=f'/workspace/ExchangeWorkspace/detections_train_detector/ArgoFiltered_GT/val_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0_withwaymovel.feather',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        detection_type='val_evaluation',
        all_car=True,
        filter_stat_before=False,
        stat_as_ignore_region=False,
        load_dir='/workspace/Argoverse2/val'))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        # ann_file=f'{rel_annotations_dir}/waymo_infos_val.pkl',
        ann_file=f'{rel_annotations_dir}/waymo_infos_train.pkl',
        pseudo_labels='/workspace/ExchangeWorkspace/detections_train_detector/ArgoFiltered_GT/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0.feather', #f'/workspace/ExchangeWorkspace/detections_train_detector/ArgoFiltered_GT/train_1.0_per_frame_remove_non_move_remove_far_filtered_version_city_w0.feather',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
        detection_type='val_detector',
        all_car=True,
        filter_stat_before=True,
        stat_as_ignore_region=False,
        load_dir='/workspace/Argoverse2/val'))

val_evaluator = dict(
    type='AV2Metric',
    ann_file=f'{data_root}/{rel_annotations_dir}/waymo_infos_train.pkl',
    # ann_file=f'{data_root}/{rel_annotations_dir}/waymo_infos_val.pkl',
    waymo_bin_file=f'{original_dataset_root}/gt.bin',
    data_root=f'{original_dataset_root}',
    backend_args=backend_args,
    convert_kitti_format=False)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
