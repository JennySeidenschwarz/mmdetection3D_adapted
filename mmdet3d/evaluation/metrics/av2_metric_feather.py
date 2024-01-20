# Copyright (c) OpenMMLab. All rights reserved.
import math
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import Config, load
from mmengine.logging import MMLogger, print_log

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, bbox3d2result,
                                points_cam2img, xywhr2xyxyr)
from .kitti_metric import KittiMetric
from scipy.spatial.transform import Rotation as R
from pyarrow import feather 
import os
import pandas as pd
import glob

import sys
sys.path.append('../')
from SeMoLi.evaluation import eval_detection
from SeMoLi.data_utils.splits import get_seq_list_fixed_val


classes = ['Car', 'Pedestrian', 'Cyclist']

column_names = [
    'log_id',
    'timestamp_ns',
    'track_uuid',
    'category',
    'length_m',
    'width_m',
    'height_m',
    'qw',
    'qx',
    'qy',
    'qz',
    'rot',
    'tx_m',
    'ty_m',
    'tz_m',
    'num_interior_pts',
    'score']

column_dtypes_dets_wo_traj = {
        'log_id': str,
    'timestamp_ns': 'int64',
    'length_m': 'float32',
    'width_m': 'float32',
    'height_m': 'float32',
    'qw': 'float32',
    'qx': 'float32',
    'qy': 'float32',
    'qz': 'float32',
    'tx_m': 'float32',
    'ty_m': 'float32',
    'tz_m': 'float32',
    'rot': 'float32',
    'num_interior_pts': 'int64',
    'score': 'float32',}


@METRICS.register_module()
class AV2MetricFeather(KittiMetric):
    """Waymo evaluation metric.

    Args:
        ann_file (str): The path of the annotation file in kitti format.
        waymo_bin_file (str): The path of the annotation file in waymo format.
        data_root (str): Path of dataset root. Used for storing waymo
            evaluation programs.
        split (str): The split of the evaluation set. Defaults to 'training'.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'mAP'.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes. Defaults to [-85, -85, -5, 85, 85, 5].
        convert_kitti_format (bool): Whether to convert the results to kitti
            format. Now, in order to be compatible with camera-based methods,
            defaults to True.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        load_type (str): Type of loading mode during training.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.
            - 'fov_image_based': Only load the instances inside the default cam
              and need to convert to the FOV-based data type to support image-
              based detector.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, KITTI: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM_FRONT'.
        use_pred_sample_idx (bool): In formating results, use the sample index
            from the prediction or from the load annotations. By default,
            KITTI: True, Waymo: False, Waymo has a conversion process, which
            needs to use the sample idx from load annotation.
            Defaults to False.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        idx2metainfo (str, optional): The file path of the metainfo in waymo.
            It stores the mapping from sample_idx to metainfo. The metainfo
            must contain the keys: 'idx2contextname' and 'idx2timestamp'.
            Defaults to None.
    """
    num_cams = 5

    def __init__(self,
                 ann_file: str = '',
                 percentage: float = 1.0,
                 detection_type: str = 'val_evaluation',
                 class_agnostic: bool = True,
                 waymo_bin_file: str = '',
                 data_root: str = '',
                 split: str = 'training',
                 metric: Union[str, List[str]] = 'mAP',
                 pcd_limit_range: List[float] = [-85, -85, -5, 85, 85, 5],
                 convert_kitti_format: bool = True,
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 pklfile_prefix: Optional[str] = None,
                 submission_prefix: Optional[str] = None,
                 load_type: str = 'frame_based',
                 default_cam_key: str = 'CAM_FRONT',
                 use_pred_sample_idx: bool = False,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None,
                 idx2metainfo: Optional[str] = None, 
                 work_dir: str = 'work_dir',
                 num_workers: int = 64,
                 filtered_file_path='',
                 flow_path='',
                 filter_static=True,
                 root_dir='') -> None:
        # self.data_infos = load(self.ann_file, percentage=self.percentage)['data_list']
        self.filtered_file_path = filtered_file_path
        self.root_dir = root_dir
        self.filter_static = filter_static
        self.flow_path = flow_path
        self.data_root = data_root
        self.num_workers = num_workers
        self.class_agnostic = class_agnostic
        if 'evaluation' in detection_type:
            self.split = 'val'
        elif 'test' in detection_type:
            self.split = 'test'
        else:
            self.split = 'train'
        self.work_dir = work_dir

        self.load_type = load_type
        self.use_pred_sample_idx = use_pred_sample_idx
        self.convert_kitti_format = convert_kitti_format
        if idx2metainfo is not None:
            self.idx2metainfo = mmengine.load(idx2metainfo)
        else:
            self.idx2metainfo = None

        super(AV2MetricFeather, self).__init__(
            ann_file=ann_file,
            percentage=percentage,
            detection_type=detection_type,
            class_agnostic=class_agnostic,
            metric=metric,
            pcd_limit_range=pcd_limit_range,
            prefix=prefix,
            pklfile_prefix=pklfile_prefix,
            submission_prefix=submission_prefix,
            default_cam_key=default_cam_key,
            collect_device=collect_device,
            backend_args=backend_args)
        self.format_only = format_only
        if self.format_only:
            assert pklfile_prefix is not None, 'pklfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleaned up at the end.'

        self.default_prefix = 'AV2 metric'
        # self.format_results_debug()

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        metric_dict = dict()
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']
        # load annotations
        self.format_results(
            results,
            classes=self.classes)
        
        return metric_dict

    def format_results(
        self,
        results: List[dict],
        classes: Optional[List[str]] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to bin file.

        Args:
            results (List[dict]): Testing results of the dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submitted files.
                It includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.

        Returns:
            tuple: (result_dict, tmp_dir), result_dict is a dict containing the
            formatted result, tmp_dir is the temporal directory created for
            saving json files when jsonfile_prefix is not specified.
        """
        torch.save(results, f'{self.work_dir}/{self.percentage}_{self.detection_type}.pth')

        '''chunk_size = math.ceil(len(results)/self.num_workers)
        self.final_results = [results[i*chunk_size:(i+1)*chunk_size] for i in range(self.num_workers)]
        print(f'Num workers {self.num_workers} and parallel processes {len(self.final_results)}')
        mmengine.track_parallel_progress(self.convert_to_argo, range(len(self.final_results)), self.num_workers)'''

        self.final_results = [results]
        for idx in range(len(self.final_results)):
            self.convert_to_argo(idx)

        # combine
        paths = glob.glob(f'{self.work_dir}/intermediate/*')
        all_df = None
        for f in paths:
            df = feather.read_feather(f)
            if all_df is None:
                all_df = df
            else:
                all_df = pd.concat([all_df, df])
            os.remove(f)
        feather.write_feather(all_df, os.path.join(self.work_dir, f'{self.percentage}_{self.detection_type}.feather'))

        torch.save(self.final_results, f'{self.work_dir}/{self.percentage}_{self.detection_type}.pth')
        print('Stored to ...', f'{self.work_dir}/{self.percentage}_{self.detection_type}.pth')

        # evaluate using our code
        self._evaluate()

    def convert_to_argo(self, idx):
        final_results = self.final_results[idx]
        argo_idx = feather.read_feather(f'{self.work_dir}/idx_to_my_idx.feather')
        os.makedirs(os.path.join(self.work_dir, 'intermediate'), exist_ok=True)
        df = pd.DataFrame(columns=column_names)
        count = 0
        for j, res in enumerate(final_results):
            # Actually, `sample_idx` here is the filename without suffix.
            # It's for identitying the sample in formating.
            # res['sample_idx'] = self.data_infos[i]['sample_idx']
            res['pred_instances_3d']['bboxes_3d'].limit_yaw(
                offset=0.5, period=np.pi * 2)
            lidar_boxes = res['pred_instances_3d']['bboxes_3d'].tensor
            scores = res['pred_instances_3d']['scores_3d']
            labels = res['pred_instances_3d']['labels_3d']
            _a2k = argo_idx[argo_idx['idx'] == res['sample_idx']]
            log_id = _a2k['log_id'].item()
            timestamp = _a2k['timestamp'].item()
            for i in range(lidar_boxes.shape[0]):
                data_row = self.parse_one_object(i, lidar_boxes, scores, labels, timestamp, log_id)
                df.loc[len(df.index)] = data_row
                if len(df) % 25000 == 0 and j != 0:
                    print('Writing to ', os.path.join(self.work_dir, 'intermediate', f'{idx}_{count}.feather'))
                    feather.write_feather(df, os.path.join(self.work_dir, 'intermediate', f'{idx}_{count}.feather'))
                    count += 1
                    df = pd.DataFrame(columns=column_names)

        print('Writing to ', os.path.join(self.work_dir, 'intermediate', f'{idx}_{count}.feather'))
        feather.write_feather(df, os.path.join(self.work_dir, 'intermediate', f'{idx}_{count}.feather'))

    def parse_one_object(self, index, lidar_boxes, scores, labels, timestamp, log_id):
        class_name = classes[labels[index].item()]

        height = lidar_boxes[index][5].item()
        heading = lidar_boxes[index][6].item()

        while heading < -np.pi:
            heading += 2 * np.pi
        while heading > np.pi:
            heading -= 2 * np.pi
        
        r = R.from_euler('z', heading)
        quat = r.as_quat()

        row = [
            log_id,
            timestamp,
            -1,
            'REGULAR_VEHICLE',#class_name,
            lidar_boxes[index][3].item(),
            lidar_boxes[index][4].item(),
            height,
            quat[3],
            quat[0],
            quat[1],
            quat[2],
            heading,
            lidar_boxes[index][0].item(),
            lidar_boxes[index][1].item(),
            lidar_boxes[index][2].item() + height / 2,
            -1,
            scores[index].item()]
        
        return row
    
    def _evaluate(self,):
        seq_list = get_seq_list_fixed_val(
            self.data_root,
            root_dir=self.root_dir,
            detection_set=self.detection_type,
            percentage=self.percentage)

        _, _, _ = eval_detection.eval_detection(
            gt_folder=self.data_root,
            trackers_folder=os.path.join(self.work_dir, f'{self.percentage}_{self.detection_type}.feather'),
            split=self.split,
            seq_to_eval=seq_list,
            remove_far=True,
            visualize=False,
            filter_class="CONVERT_ALL_TO_CARS" if self.class_agnostic else "NO_FILTER",
            filter_moving=self.filter_static,
            min_num_interior_pts=0,
            root_dir=self.root_dir,
            filtered_file_path=self.filtered_file_path,
            flow_path=self.flow_path)
