# Copyright (c) OpenMMLab. All rights reserved.
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError('Please run "pip install waymo-open-dataset-tf-2-6-0" '
                      '>1.4.5 to install the official devkit first.')

import os.path as osp
import os
import glob
from collections import defaultdict
import pandas as pd
from typing import Callable, List, Union
import tensorflow as tf

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset
from .kitti_dataset import KittiDataset 
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from pyarrow import feather


@DATASETS.register_module()
class AV2Dataset(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        data_prefix (dict): data prefix for point cloud and
            camera data dict. Defaults to dict(
                                    pts='velodyne',
                                    CAM_FRONT='image_0',
                                    CAM_FRONT_LEFT='image_1',
                                    CAM_FRONT_RIGHT='image_2',
                                    CAM_SIDE_LEFT='image_3',
                                    CAM_SIDE_RIGHT='image_4')
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used
            as input. Defaults to dict(use_lidar=True).
        default_cam_key (str): Default camera key for lidar2img
            association. Defaults to 'CAM_FRONT'.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (List[float]): The range of point cloud
            used to filter invalid predicted boxes.
            Defaults to [-85, -85, -5, 85, 85, 5].
        cam_sync_instances (bool): If use the camera sync label
            supported from waymo version 1.3.1. Defaults to False.
        load_interval (int): load frame interval. Defaults to 1.
        max_sweeps (int): max sweep for each frame. Defaults to 0.
    """
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist'),
        'palette': [
            (0, 120, 255),  # Waymo Blue
            (0, 232, 157),  # Waymo Green
            (255, 205, 85)  # Amber
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: dict = dict(
                     pts='velodyne',
                     CAM_FRONT='image_0',
                     CAM_FRONT_LEFT='image_1',
                     CAM_FRONT_RIGHT='image_2',
                     CAM_SIDE_LEFT='image_3',
                     CAM_SIDE_RIGHT='image_4'),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM_FRONT',
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 cam_sync_instances: bool = False,
                 load_interval: int = 1,
                 max_sweeps: int = 0,
                 pseudo_labels = None,
                 load_dir='/workspace/waymo/waymo_format/training',
                 filter_empty_3dboxes=True,
                 only_matched=False,
                 stat_as_ignore_region=False,
                 filter_stat_before=False,
                 **kwargs) -> None:
        self.stat_as_ignore_region = stat_as_ignore_region
        self.filter_stat_before = filter_stat_before
        self.load_interval = load_interval
        # set loading mode for different task settings
        self.cam_sync_instances = cam_sync_instances
        # construct self.cat_ids for vision-only anns parsing
        self.cat_ids = range(len(self.METAINFO['classes']))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.max_sweeps = max_sweeps
        self.pseudo_labels = pseudo_labels
        self.only_matched = only_matched
        self.filter_empty_3dboxes = filter_empty_3dboxes
        self.tfrecord_pathnames = sorted(
            glob.glob(osp.join(load_dir, '*.tfrecord')))
        self.load_dir = load_dir
        self._class_dict_argo = {
            -1: 'UNMATCHED',
            1: 'REGULAR_VEHICLE',
            2: 'PEDESTRIAN',
            3: 'BOLLARD',
            4: 'CONSTRUCTION_CONE',
            5: 'CONSTRUCTION_BARREL',
            6: 'STOP_SIGN',
            7: 'BICYCLE',
            8: 'LARGE_VEHICLE',
            9: 'WHEELED_DEVICE',
            10: 'BUS',
            11: 'BOX_TRUCK',
            12: 'SIGN',
            13: 'TRUCK',
            14: 'MOTORCYCLE',
            15: 'BICYCLIST',
            16: 'VEHICULAR_TRAILER',
            17: 'TRUCK_CAB',
            18: 'MOTORCYCLIST',
            19: 'DOG',
            20: 'SCHOOL_BUS',
            21: 'WHEELED_RIDER',
            22: 'STROLLER',
            23: 'ARTICULATED_BUS',
            24: 'MESSAGE_BOARD_TRAILER',
            25: 'MOBILE_PEDESTRIAN_SIGN',
            26: 'WHEELCHAIR',
            27: 'RAILED_VEHICLE',
            28: 'OFFICIAL_SIGNALER',
            29: 'TRAFFIC_LIGHT_TRAILER',
            30: 'ANIMAL',
            31: 'MOBILE_PEDESTRIAN_CROSSING_SIGN'}
        
        self.selected_argo_classes = ['REGULAR_VEHICLE'] #[v for k, v in self._class_dict_argo.items() if not k in [-1, 3, 4, 5, 6, 31]]

        self.argo_to_int = {c: i for i, c in enumerate(self.selected_argo_classes)}
        
        self.argo_to_kitti = {'REGULAR_VEHICLE': 'Car'}

        # we do not provide backend_args to custom_3d init
        # because we want disk loading for info
        # while ceph loading for Prediction2Waymo
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            pcd_limit_range=pcd_limit_range,
            default_cam_key=default_cam_key,
            data_prefix=data_prefix,
            test_mode=test_mode,
            load_type=load_type,
            **kwargs)

    def get_pose(self, log_id, timestamp_ns):
        log_poses_df = feather.read_feather(os.path.join(self.load_dir, log_id, "city_SE3_egovehicle.feather"))
        pose_df = log_poses_df.loc[log_poses_df["timestamp_ns"] == timestamp_ns]
        qw, qx, qy, qz = pose_df[["qw", "qx", "qy", "qz"]].to_numpy().squeeze()
        tx_m, ty_m, tz_m = pose_df[["tx_m", "ty_m", "tz_m"]].to_numpy().squeeze()
        city_t_ego = np.array([tx_m, ty_m, tz_m])
        r = R.from_quat([qx, qy, qz, qw])
        rot = r.as_matrix()
        pose = np.zeros([4, 4])
        pose[:3, :3] = rot
        pose[:3, -1] = city_t_ego
        return pose

    def _parse_ann_info(self, info, info_pkl, log_id) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_bboxes_labels, centers_2d, depths, ignore_bbs3d = \
            self.get_bbs(info, log_id)
        # in waymo, lidar2cam = R0_rect @ Tr_velo_to_cam
        # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes=gt_bboxes,
            gt_bboxes_labels=gt_bboxes_labels,
            centers_2d=centers_2d,
            depths=depths)
        return anns_results

    def get_bbs(self, dets, log_id):
        types = list()
        bbs3d = list()
        ignore_bbs3d = list()
        for ids, obj in dets.iterrows(): 
            my_type = obj['category']
            
            if my_type not in self.selected_argo_classes:
                continue
            if self.filter_empty_3dboxes and obj['num_interior_pts'] < 1 and obj['num_interior_pts'] != -1:
                continue
            
            my_type = self.label_mapping[self.METAINFO['classes'].index(self.argo_to_kitti[my_type])]
            
            height = obj['height_m']
            width = obj['width_m']
            length = obj['length_m']
            
            x = obj['tx_m']
            y = obj['ty_m']
            z = obj['tz_m'] - height / 2
            
            # qw = torch.cos(det.heading/2)
            r = R.from_quat([obj['qx'], obj['qy'], obj['qz'], obj['qw']])
            rotation_y = r.as_euler('xyz')[-1]
            
            '''
            # project bounding box to the virtual reference frame
            T_velo_to_front_cam = self._get_T_velo_to_front_cam(log_id, frame_idx)
            pt_ref = T_velo_to_front_cam @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            bounding_box_3d = np.array([[
                round(x, 2),
                round(y, 2),
                round(z, 2),
                round(length, 2),
                round(height, 2),
                round(width, 2),
                round(rotation_y, 2)]], dtype=np.float32)
            '''
            bounding_box_3d = np.array([[
                round(x, 2),
                round(y, 2),
                round(z, 2),
                round(length, 2),
                round(width, 2),
                round(height, 2),
                round(rotation_y, 2)]], dtype=np.float32)
            
            if self.stat_as_ignore_region and 'filter_moving' in dets.columns:
                if not obj['filter_moving']:
                    ignore_bbs3d.append(bounding_box_3d)
                    continue
            
            if my_type != -1:
                cat_name = self.metainfo['classes'][my_type]
                self.num_ins_per_cat[cat_name] += 1
            else:
                continue
            
            bbs3d.append(bounding_box_3d)
            types.append(np.array(my_type, dtype=np.int64))
        
        # if no bbs
        if len(bbs3d) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros((dets.shape[0], 4), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float32), np.zeros((0), dtype=np.float32), np.zeros((0, 7), dtype=np.float32)

        bbs3d = np.vstack(bbs3d)
        types = np.stack(types)
        if len(ignore_bbs3d):
            ignore_bbs3d = np.vstack(ignore_bbs3d)
        else:
            ignore_bbs3d = np.zeros((0, 7), dtype=np.float32)
        
        return bbs3d, types, np.zeros((dets.shape[0], 4), dtype=np.float32), np.zeros((dets.shape[0]), dtype=np.int64), np.zeros((dets.shape[0], 2), dtype=np.float32), np.zeros((dets.shape[0]), dtype=np.float32), ignore_bbs3d

    def load_data_list(self, detection_type, ann_file2=False) -> List[dict]:
        """Add the load interval."""
        if self.pseudo_labels is None or ann_file2:
            data_list = super().load_data_list(detection_type=detection_type)
        else:
            if not os.path.isfile(self.pseudo_labels):
                self.pseudo_labels = os.path.join(self.pseudo_labels, self.detection_type)
            self.tfrecord_pathnames = {
                p.split('/')[-1].split('-')[1].split('_')[0]: p for p in self.tfrecord_pathnames}
            self.T_velo_to_front_cam_dict = defaultdict(dict)
            if os.path.isfile('T_velo_to_front_cam_dict_train.npz'):
                self.T_velo_to_front_cam_dict = np.load(
                        'T_velo_to_front_cam_dict_train.npz', allow_pickle=True)['T_velo_to_front_cam_dict'].item()

            with open(f'/workspace/ExchangeWorkspace/new_seq_splits_AV2_fixed_val//{self.percentage}_{self.detection_type}.txt', 'r') as f:
                self.seqs = f.read()
                self.seqs = self.seqs.split('\n')
                self.seqs = [s for s in self.seqs]
            if self.ann_file2 is not '' and self.ann_file2[-3:] != 'pkl':
                with open(f'/workspace/ExchangeWorkspace/new_seq_splits_AV2_fixed_val/{self.percentage}_train_gnn.txt', 'r') as f:
                    seqs2 = f.read()
                    seqs2 = seqs2.split('\n')
                    seqs2 = [s for s in seqs2]
                    self.seqs = self.seqs + seqs2
            data_list = self._load_data_list()
        data_list = data_list[::self.load_interval]
        return data_list
    
    def _load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        
        metainfo = {'categories': self.argo_to_int, 'dataset': 'argoverse', 'version': '2', 'info_version': '2'}

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)
        
        # LOAD PSEUDO LABELS
        from pyarrow import feather
        if os.path.isfile(self.pseudo_labels):
            raw_data_list = feather.read_feather(os.path.join(self.pseudo_labels))
        else:
            for i, f in enumerate(os.listdir(self.pseudo_labels)):
                if i == 0:
                    raw_data_list = feather.read_feather(os.path.join(self.pseudo_labels, f, 'annotations.feather'))
                else:
                    raw_data_list = raw_data_list.append(feather.read_feather(os.path.join(self.pseudo_labels, f, 'annotations.feather')))
        
        raw_data_list = raw_data_list.astype({'timestamp_ns': int})
        if raw_data_list['category'].dtype == int:
            def convert2int(x): return self._class_dict_argo[x]
            raw_data_list['category'] = raw_data_list['category'].apply(convert2int)
        print(f'All labels {raw_data_list.shape[0]}')
        print(self.ann_file2 is not '' and self.ann_file2[-3:] != 'pkl')
        if self.ann_file2 is not '' and self.ann_file2[-3:] != 'pkl':
            
            raw_data_list['filter_moving'] = True
            print(f'Loading ann_file2 {self.ann_file2}')
            if os.path.isfile(self.ann_file2):
                raw_data_list2 = feather.read_feather(os.path.join(self.ann_file2))
            else:
                for i, f in enumerate(os.listdir(self.ann_file2)):
                    if i == 0:
                        raw_data_list2 = feather.read_feather(os.path.join(self.ann_file2, f, 'annotations.feather'))
                    else:
                        raw_data_list2 = raw_data_list.append(feather.read_feather(os.path.join(self.ann_file2, f, 'annotations.feather')))
            raw_data_list2 = raw_data_list2.astype({'timestamp_ns': int})
            if raw_data_list2['category'].dtype == int:
                def convert2int(x): return self._class_dict_argo[x]
                raw_data_list2['category'] = raw_data_list2['category'].apply(convert2int)
            raw_data_list = raw_data_list.append(raw_data_list2)
            print(f'Labels of both sources {raw_data_list.shape[0]}')

        seqs = [str(s) for s in self.seqs]
        raw_data_list = raw_data_list[raw_data_list['log_id'].isin(seqs)]
        print(f'Labels of valid sequences {raw_data_list.shape[0]}')
        raw_data_list = raw_data_list.astype({'log_id': str})
        if self.only_matched:
            print(f'All detections {raw_data_list.shape[0]}')
            raw_data_list = raw_data_list[raw_data_list['matched_category'] != 'TYPE_UNKNOWN']
            print(f'Only matched detections {raw_data_list.shape[0]}')
        
        if self.filter_stat_before and 'filter_moving' in raw_data_list.columns:
            raw_data_list = raw_data_list[raw_data_list['filter_moving']]

        raw_data_list = raw_data_list[raw_data_list['log_id'].isin(self.seqs)]
        raw_data_list = raw_data_list[raw_data_list['category'].isin(self.selected_argo_classes)] 
        
        print(f'Number of detections without Sign {raw_data_list.shape[0]}')

        if self.all_car:
            raw_data_list['category'] = 'REGULAR_VEHICLE'
        
        raw_data_list = raw_data_list[raw_data_list['log_id'].isin([str(s) for s in self.seqs])]
        raw_data_list['lidar_path'] = np.ones(raw_data_list.shape[0])

        # load and parse data_infos.
        data_list = []
        num_ids = len(raw_data_list['log_id'].unique())
        count = 0
        idx_to_my_idx = pd.DataFrame(columns=['idx', 'log_id', 'timestamp'])
        for i, log_id in enumerate(raw_data_list['log_id'].unique()):
            log_data = raw_data_list[raw_data_list['log_id'] == log_id]
            num_time = len(log_data['timestamp_ns'].unique())
            print(f'{i} / {num_ids}, in total {num_time} timestamps')
            for time in log_data['timestamp_ns'].unique():
                raw_data_info_pkl = dict()
                raw_data_info_pkl['sample_idx'] = count
                idx_to_my_idx.loc[len(idx_to_my_idx.index)] = [count, log_id, time]
                count += 1
                raw_data_info_pkl['sample_idx_mine'] = f'{log_id}_{time}'
                raw_data_info_pkl['timestamp'] = time
                raw_data_info_pkl['ego2global'] = self.get_pose(log_id, time)
                #raw_data_info_pkl['images'] = []
                #raw_data_info_pkl['cam_sync_instances'] = []
                #raw_data_info_pkl['cam_instances'] = []

                raw_data_info = log_data[log_data['timestamp_ns'] == time]
                lidar_path = os.path.join(self.load_dir, log_id, 'sensors/lidar', f'{time}.feather')
                raw_data_info_pkl['lidar_points'] = dict()
                raw_data_info_pkl['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix.get('pts', ''), lidar_path)
                if not os.path.isfile(raw_data_info_pkl['lidar_points']['lidar_path']):
                    print('Oh no...')
                    continue
                raw_data_info_pkl['lidar_points']['num_pts_feats'] = 4
                raw_data_info_pkl['lidar_path'] = osp.join(
                            self.data_prefix.get('pts', ''), lidar_path)
                raw_data_info_pkl['num_pts_feats'] = 4

                # if time == 1507221646275518:
                # parse raw data information to target format
                data_info = self._parse_data_info(
                    raw_data_info, raw_data_info_pkl, log_id)
                
                if isinstance(data_info, dict):
                    # For image tasks, `data_info` should information if single
                    # image, such as dict(img_path='xxx', width=360, ...)
                    data_list.append(data_info)
                elif isinstance(data_info, list):
                    # For video tasks, `data_info` could contain image
                    # information of multiple frames, such as
                    # [dict(video_path='xxx', timestamps=...),
                    #  dict(video_path='xxx', timestamps=...)]
                    for item in data_info:
                        if not isinstance(item, dict):
                            raise TypeError('data_info must be list of dict, but '
                                            f'got {type(item)}')
                    data_list.extend(data_info)
                else:
                    raise TypeError('data_info should be a dict or list of dict, '
                                    f'but got {type(data_info)}')
        feather.write_feather(idx_to_my_idx, 'idx_to_my_idx.feather')
        return data_list
    
    def _parse_data_info(self, info, info_pkl, log_id) -> Union[dict, List[dict]]:
        """if task is lidar or multiview det, use super() method elif task is
        mono3d, split the info from frame-wise to img-wise."""
        # print(info)
        # print(info[['tx_m', 'ty_m', 'tz_m', 'qw', 'qz', 'qy', 'qx', 'length_m', 'width_m', 'height_m']])
        # print(info_pkl['ann_info'])
        if not self.test_mode:
            # used in training
            info_pkl['ann_info'] = self._parse_ann_info(info, info_pkl, log_id)
        if self.test_mode and self.load_eval_anns:
            info_pkl['eval_ann_info'] = self._parse_ann_info(info, info_pkl, log_id)
        # print(info_pkl['ann_info'])
        return info_pkl

