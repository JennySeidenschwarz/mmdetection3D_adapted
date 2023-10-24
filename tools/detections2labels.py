### kitti_dbinfos_train.pkl = {'c1': [], 'c2': [], ...}
# [] = [{'name': 'Pedestrian', 'path': 'kitti_gt_database/0_Pedestrian_0.bin', 'image_idx': 0, 'gt_idx': 0, 'box3d_lidar': array([ 8.731381 , -1.8559176, -1.5996994,  1.2      ,  0.48     ,1.89     , -1.5807964], dtype=float32), 'num_points_in_gt': 377, 'difficulty': 0, 'group_id': 0, 'score': 0.0}, {}, {}, ...]

### kitti_infos_{split}.pkl = {'metainfo': {}, 'data_list': []}
# [] = [{['sample_idx': int, 'images': [], 'lidar_points': [], 'instances': [], 'plane': np.array, 'cam_instances': {}]}]
# 'instances' = [{'bbox': [712.4, 143.0, 810.73, 307.92], 'bbox_label': 0, 'bbox_3d': [1.84, 1.47, 8.41, 1.2, 1.89, 0.48, 0.01], 'bbox_label_3d': 0, 'depth': 8.4149808883667, 'center_2d': [763.7633056640625, 224.4706268310547], 'num_lidar_pts': 377, 'difficulty': 0, 'truncated': 0.0, 'occluded': 0, 'alpha': -0.2, 'score': 0.0, 'index': 0, 'group_id': 0}]


# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError('Please run "pip install waymo-open-dataset-tf-2-6-0" '
                      '>1.4.5 to install the official devkit first.')

import os
from glob import glob
from os.path import exists, join

import mmengine
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import \
    parse_range_image_and_camera_projection
from pyarrow import feather
import os.path as osp
from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
from tools.dataset_converters.create_gt_database import GTDatabaseCreater
import argparse
import pandas as pd
from mmdet3d.utils import replace_ceph_backend
split_names = {'train': 'training', 'val': 'validation', 'test': 'testing'}
split_names_reverse = {v: k for k, v in split_names.items()}


class Argoverse22Kitti(object):
    """Argoverse2 to Waymo converter.

    This class serves as the converter to change the argoverse 2 labels to
    waymo format.

    Args:
        load_dir (str): Directory to load argoverse detections.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
        test_mode (bool, optional): Whether in the test_mode.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 dets_dir,
                 prefix,
                 workers=64,
                 test_mode=False,
                 save_cam_sync_labels=False,
                 per=1.0,
                 matched_gt=None,
                 min_num_interior=20):
        self.id = 0
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True
        self.min_num_interior = min_num_interior
        
        self.selected_waymo_classes = ['REGULAR_VEHICLE', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        # keep the order defined by the official protocol
        self.cam_list = [
            '_FRONT',
            '_FRONT_LEFT',
            '_FRONT_RIGHT',
            '_SIDE_LEFT',
            '_SIDE_RIGHT',
        ]
        self.lidar_list = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
        self.type_list = [
            'UNKNOWN', 'REGULAR_VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'REGULAR_VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }

        self.load_dir = load_dir
        
        if matched_gt == 'IoU3D':
            dets_dir = dets_dir + '/annotations_IoU3D.feather'
        elif matched_gt == 'Center':
            dets_dir = dets_dir + '/annotations_CENTER.feather'

        self.save_dir = save_dir
        if os.path.isfile(dets_dir):
            self.dets = feather.read_feather(dets_dir)
            print(f'Before filtering {self.dets.shape}...')
            self.dets = self.dets[self.dets['matched_category'] != 'UNMATCHED']
            print(f'After filtering {self.dets.shape}...')
            self.log_ids_dets = self.dets['log_id'].unique()
        else:
            self.log_ids_dets = os.listdir(dets_dir)
        self.dets_dir = dets_dir
        
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode
        self.save_cam_sync_labels = save_cam_sync_labels

        self.tfrecord_pathnames = sorted(
            glob(join(self.load_dir, '*.tfrecord')))

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        if self.save_cam_sync_labels:
            self.cam_sync_label_save_dir = f'{self.save_dir}/cam_sync_label_'
            self.cam_sync_label_all_save_dir = \
                f'{self.save_dir}/cam_sync_label_all'
        self.per = per
        
        if self.prefix == 0:
            with open(f'new_seq_splits_Waymo_Converted_fixed_val/{self.per}_train_detector.txt', 'r') as f:
                self.seqs = f.read()
                self.seqs = self.seqs.split('\n')

            self.waymo2kitti = pd.read_csv(f'logid_2_kitti/{self.prefix}_naming.csv')

        elif self.prefix == 1:
            with open(f'new_seq_splits_Waymo_Converted_fixed_val/{self.per}_val_detector.txt', 'r') as f:
                self.seqs = f.read()
                self.seqs = self.seqs.split('\n')

            self.waymo2kitti = pd.read_csv(f'logid_2_kitti/{self.prefix}_naming.csv')

        else:
            self.seqs = None
            self.waymo2kitti = None

        if self.seqs:
            self.waymo2kitti['path'] = self.waymo2kitti['log_id']
            # self.waymo2kitti['log_id'] = [p.split('/')[-1].split('-')[1].split('_')[0] for p in self.waymo2kitti['log_id']]
            self.waymo2kitti['log_id'] = self.waymo2kitti['log_id'].astype(str)
            self.waymo2kitti = self.waymo2kitti[self.waymo2kitti['log_id'].isin(self.seqs)]
            self.tfrecord_pathnames_filtered = [p for p in sorted(
                glob(join(self.load_dir, '*.tfrecord'))) \
                        if p.split('/')[-1].split('-')[1].split('_')[0] in self.waymo2kitti['log_id'].values.tolist()]

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        print(len(self), self.selected_waymo_classes)
        mmengine.track_parallel_progress(self.convert_one, range(len(self)),
                                         self.workers)
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        if pathname not in self.tfrecord_pathnames_filtered:
            print(f'Path not in current detection set {pathname}...')
            return
        print(f'Converting {pathname} of total {len(self.tfrecord_pathnames_filtered)} files...')
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')
        log_id = pathname.split('-')[1].split('_')[0]
        
        if log_id not in self.log_ids_dets:
            return
        
        if os.path.isfile(self.dets_dir):
            dets = self.dets[self.dets['log_id'] == log_id]
        else:
            dets = feather.read_feather(self.dets_dir + '/' + log_id + '/' + 'annotations.feather')
        
        for frame_idx, data in enumerate(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            _dets = dets[dets['timestamp_ns'] == frame.timestamp_micros]
            if (self.selected_waymo_locations is not None
                    and frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue

            self.get_T_velo_to_front_cam(frame)
            
            if not self.test_mode:
                # TODO save the depth image for waymo challenge solution.
                self.save_label(frame, file_idx, frame_idx, _dets, log_id)
                if self.save_cam_sync_labels:
                    self.save_label(frame, file_idx, frame_idx, cam_sync=True)

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def get_T_velo_to_front_cam(self, frame):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
                break

    def save_label(self, frame, file_idx, frame_idx, dets, log_id, cam_sync=False):
        """Parse and save the label data in txt format.
        The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
            cam_sync (bool, optional): Whether to save the cam sync labels.
                Defaults to False.
        """
        label_all_path = f'{self.label_all_save_dir}/{self.prefix}' + \
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'
        fp_label_all = open(label_all_path, 'w+')
        for ids, obj in dets.iterrows():
            # NOTE: the 2D labels do not have strict correspondence with
            # the projected 2D lidar labels
            # e.g.: the projected 2D labels can be in camera 2
            # while the most_visible_camera can have id 4

            name = '0'
            bounding_box = (0, 0, 0, 0)

            my_type = obj['category']
            
            if my_type not in self.selected_waymo_classes:
                continue

            if self.filter_empty_3dboxes and int(obj['num_interior_pts']) < self.min_num_interior:
                continue

            my_type = self.waymo_to_kitti_class_map[my_type]

            height = obj['height_m']
            width = obj['width_m']
            length = obj['length_m']

            x = obj['tx_m']
            y = obj['ty_m']
            z = obj['tz_m'] - height / 2
            
            # project bounding box to the virtual reference frame
            pt_ref = self.T_velo_to_front_cam @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()
            
            # qw = torch.cos(det.heading/2)
            # rotation_y = -np.arccos(obj['qw'])*2 - np.pi / 2
            rotation_y = -obj['rot'] - np.pi / 2
            track_id = obj['gt_id'] if 'gt_id' in dets.columns else log_id + '_' + str(ids)

            # not available
            truncated = 0
            occluded = 0
            alpha = -10
            # Car 0 0 -10 0 0 0 0 1.81 2.5 5.8 -3.64 1.89 -11.59 nan 0
            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))
            
            if self.save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            label_path = f'{self.label_save_dir}{name}/{self.prefix}' + \
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'
            fp_label = open(label_path, 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all) 
        fp_label_all.close()

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir,
            ]
            dir_list2 = [self.label_save_dir]
            if self.save_cam_sync_labels:
                dir_list1.append(self.cam_sync_label_all_save_dir)
                dir_list2.append(self.cam_sync_label_save_dir)
        else:
            dir_list1 = []
            dir_list2 = []
        for d in dir_list1:
            mmengine.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmengine.mkdir_or_exist(f'{d}{str(i)}')

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret


def create_ImageSets_img_ids(root_dir, splits, waymo2kitti):
    save_dir = join(root_dir, 'ImageSets/')
    if not exists(save_dir):
        os.mkdir(save_dir)
    idx_all = [[] for i in splits]
    for i, split in enumerate(splits):
        path = join(root_dir, splits[i], 'label_all')
        if not exists(path):
            RawNames = []
        else:
            RawNames = os.listdir(path)

        if split in waymo2kitti.keys():
            waymo2kitti[split]['prefix'] = waymo2kitti[split]['prefix'].apply(lambda x: str(x))
            make_str = lambda x: f'{str(x).zfill(3)}'
            waymo2kitti[split]['frame_idx'] = waymo2kitti[split]['frame_idx'].apply(make_str)
            waymo2kitti[split]['file_idx'] = waymo2kitti[split]['file_idx'].apply(make_str)
            waymo2kitti[split]['whole_name'] = waymo2kitti[split]['prefix'] + waymo2kitti[split]['file_idx'] + waymo2kitti[split]['frame_idx']
            to_use_list = waymo2kitti[split]['whole_name'].values.tolist()
        
        for j, name in enumerate(RawNames):
            if j % 1000 == 0:
                print(f'{j}/{len(RawNames)}')
            if name.endswith('.txt'):
                idx = name.replace('.txt', '\n')
                to_use = idx.strip('\n') in to_use_list if split in waymo2kitti.keys() else True
                if int(idx[0]) <= i:
                    idx_all[int(idx[0])].append(idx)
        
        idx_all[i].sort()
    
    open(save_dir + 'train.txt', 'w').writelines(idx_all[0])
    # open(save_dir + 'val.txt', 'w').writelines(idx_all[1])
    # open(save_dir + 'trainval.txt', 'w').writelines(idx_all[0] + idx_all[1])
    # open(save_dir + 'test.txt', 'w').writelines(idx_all[2])
    # open(save_dir+'test_cam_only.txt','w').writelines(idx_all[3])
    print('created txt files indicating what to collect in ', splits)

def create_info_files(out_dir, splits, info_prefix, max_sweeps, workers, calib_dir, waymo2kitti, detection_set):
    
    # from tools.dataset_converters.waymo_converter import \
    #     create_ImageSets_img_ids
    # # create_ImageSets_img_ids(out_dir, list(split_names_reverse.keys()), calib_dir, waymo2kitti)
    create_ImageSets_img_ids(out_dir, splits, waymo2kitti)
    
    # Generate waymo infos
    kitti.create_waymo_info_file(
        out_dir, info_prefix, save_path=out_dir, sensor_path=calib_dir, max_sweeps=max_sweeps, workers=workers, label_path=out_dir, cam_sync_labels=False)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    # info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    # info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    # info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    print('UPDATING')
    update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_train_path)
    # update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_val_path)
    # update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_trainval_path)
    # update_pkl_infos('waymo', out_dir=out_dir, pkl_path=info_test_path)
     
    GTDatabaseCreater(
            'WaymoDataset',
            '/workspace/waymo_kitti_format/kitti_format',
            info_prefix,
            f'{info_prefix}_infos_train.pkl',
            relative_path=False,
            with_mask=False,
            num_worker=workers,
            out_path=out_dir).create(detection_type=detection_set)


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--waymo_format_dir',
        type=str,
        default='/workspace/waymo/waymo_format/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--waymo_kitti_format_anno_dir',
        type=str,
        default='/workspace/waymo_kitti_format_annotations/kitti_format/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--waymo_kitti_format_dir',
        type=str,
        default='/workspace/waymo_kitti_format/kitti_format/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0',
        required=False,
        help='specify the dataset version, no need for kitti')
    parser.add_argument(
        '--max-sweeps',
        type=int,
        default=10,
        required=False,
        help='specify sweeps of lidar per example')
    parser.add_argument(
        '--kitti_dets_dir',
        type=str,
        default='data/pseudo_labels/kitti_format/', #'/workspace/result/kitti_format/',
        required=False,
        help='name of info pkl')
    parser.add_argument(
        '--av2_dets_dir',
        type=str,
        default='detections_50_2/all_egocomp_margin0.6_width25_flow_0.1_10/',
        required=False,
        help='name of info pkl')
    parser.add_argument('--extra-tag', type=str, default='waymo')
    parser.add_argument(
        '--workers', type=int, default=64, help='number of threads to be used')
    parser.add_argument(
        '--convert',
        action='store_true',
        help='Whether to convert AV2 detections to kitti format.')
    parser.add_argument(
        '--update_pkl',
        action='store_true',
        help='Wheather to update pkl info files.')
    parser.add_argument('--detection_type')
    parser.add_argument('--percentage')
    parser.add_argument('--detection_file')
    parser.add_argument('--matched_gt', default=None)
    parser.add_argument('--min_num_interior', type=int)

    return parser.parse_args()


if __name__ == "__main__":
    from mmdet3d.utils import register_all_modules
    register_all_modules()
    args = parse_args()
    waymo2kitti = dict()
    if 'evaluation' in args.detection_type:
        split = 'evaluation'
        prefix = 1
    else:
        split = 'training'
        prefix = 0
    per = args.percentage
    det_split = args.detection_type
    
    save_dir = args.kitti_dets_dir + args.detection_file + f'_{args.min_num_interior}'

    if args.matched_gt == 'IoU3D':
        save_dir = save_dir + '_IoU3D'
    elif args.matched_gt == 'Center':
        save_dir = save_dir + '_CENTER'

    converter = Argoverse22Kitti(args.waymo_format_dir + split,
                save_dir + '/' + split,
                args.av2_dets_dir + args.detection_file + '/' + det_split,
                prefix,
                args.workers, 
                per=per,
                matched_gt=args.matched_gt,
                min_num_interior=args.min_num_interior)
    if args.convert:
        converter.convert()
    print('add', split)
    waymo2kitti[split] = converter.waymo2kitti
        
    if args.update_pkl:
        splits = ['training']
        create_info_files(
                save_dir + '/',
                splits,
                args.extra_tag + os.path.basename(args.av2_dets_dir),
                args.max_sweeps,
                args.workers,
                args.waymo_kitti_format_dir,
                waymo2kitti,
                detection_set=det_split)

