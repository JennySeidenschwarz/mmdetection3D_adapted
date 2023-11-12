# Copyright (c) OpenMMLab. All rights reserved.
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


@METRICS.register_module()
class AV2Metric(KittiMetric):
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
                 ann_file: str,
                 percentage: float,
                 detection_type: str,
                 all_car: bool,
                 waymo_bin_file: str,
                 data_root: str,
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
                 work_dir: str = 'work_dir') -> None:
        # self.data_infos = load(self.ann_file, percentage=self.percentage)['data_list']
        self.waymo_bin_file = waymo_bin_file
        self.data_root = data_root
        if 'evaluation' in detection_type:
            self._split = 'validation'
        elif 'test' in detection_type:
            self._split = 'testing'
        else:
            self._split = 'training'
            all_car_add = '_car' if all_car else ''
            self.waymo_bin_file = f'waymo_gt_and_meta/gt/gt_{percentage}_{detection_type}{all_car_add}.bin'
            self.waymo_bin_file = '/workspace/ExchangeWorkspace/waymo_gt_and_meta/gt/gt_0.1_val_detector_car_filter_moving_range.bin'
        self.work_dir = work_dir

        print('GT TO EVALUATEEEEEEEEEEEE', self.waymo_bin_file, self.work_dir, 'hiiii')

        self.split = split
        self.load_type = load_type
        self.use_pred_sample_idx = use_pred_sample_idx
        self.convert_kitti_format = convert_kitti_format
        if idx2metainfo is not None:
            self.idx2metainfo = mmengine.load(idx2metainfo)
        else:
            self.idx2metainfo = None

        super(AV2Metric, self).__init__(
            ann_file=ann_file,
            percentage=percentage,
            detection_type=detection_type,
            all_car=all_car,
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

        self.default_prefix = 'Waymo metric'
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
        final_results = results
        for i, res in enumerate(final_results):
            # Actually, `sample_idx` here is the filename without suffix.
            # It's for identitying the sample in formating.
            # res['sample_idx'] = self.data_infos[i]['sample_idx']
            res['pred_instances_3d']['bboxes_3d'].limit_yaw(
                offset=0.5, period=np.pi * 2)

        torch.save(final_results, f'{self.work_dir}/{self.percentage}_{self.detection_type}.pth')
        print('Stored to ...', f'{self.work_dir}/{self.percentage}_{self.detection_type}.pth')



