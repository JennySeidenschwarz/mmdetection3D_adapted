import shutil
import argparse
# from __future__ import print_function
from pyarrow import feather
import os
from waymo_open_dataset.protos import metrics_pb2
import tensorflow as tf
from collections import defaultdict
from scipy.spatial.transform import Rotation
import pandas as pd
import glob
import shutil
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch


'''
WAYMO CLASSES:
1 = VEHICLE
2 = PEDESTRIAN
3 = SIGN
4 = CYCLIST
'''

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



def get_waymo2kitti(detection_type, percentage):
    if 'evaluation' not in detection_type:
        waymo2kitti = pd.read_csv(f'logid_2_kitti_w_time/0_naming.csv')
    else:
        waymo2kitti = pd.read_csv(f'logid_2_kitti_w_time/1_naming.csv')

    with open(f'new_seq_splits_Waymo_Converted_fixed_val/{percentage}_{detection_type}.txt', 'r') as f:
        seqs = f.read()
        seqs = seqs.split('\n')
        seqs = [int(s) for s in seqs]
    waymo2kitti = waymo2kitti[waymo2kitti['log_id'].isin(seqs)]
    waymo2kitti['prefix'] = waymo2kitti['prefix'].apply(lambda x: str(x))
    make_str = lambda x: f'{str(x).zfill(3)}'
    waymo2kitti['frame_idx'] = waymo2kitti['frame_idx'].apply(make_str)
    waymo2kitti['file_idx'] = waymo2kitti['file_idx'].apply(make_str)
    waymo2kitti['whole_name'] = waymo2kitti['prefix'] + waymo2kitti['file_idx'] + waymo2kitti['frame_idx']

    return waymo2kitti

def main(file, save_dir, detection_type, percentage, argo=False):
    detections = torch.load(file)
    save_path = os.path.join(save_dir, file.split('/')[1])
    file_name = file.split('/')[2][:-3] + 'feather'

    waymo2kitti = get_waymo2kitti(detection_type, percentage)
    os.makedirs(os.path.join(save_dir, file.split('/')[1]), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'intermediate'), exist_ok=True)
     
    convert(detections, waymo2kitti, save_path, os.path.dirname(file), argo=argo)

    paths = glob.glob(f'{save_path}/intermediate/*')
    all_df = None
    for f in paths:
        df = feather.read_feather(f)
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df])
        os.remove(f)

    feather.write_feather(all_df, os.path.join(save_path, file_name))
        
    print(f"Stored to detections converted from {file} to {save_path}")


def parse_one_object(index, lidar_boxes, scores, labels, timestamp, log_id):
    try:
        class_name = classes[labels[index].item()]
    except:
        print(lidar_boxes.shape, index, labels.shape, scores.shape)
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
        'REGULAR_VEHICLE', #class_name,
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

def convert(results, waymo2kitti, save_path, idx_path, argo=False):
    df = pd.DataFrame(columns=column_names)
    if argo:
        argo_idx = feather.read_feather(f'{idx_path}/idx_to_my_idx.feather')
    else:
        waymo2kitti = waymo2kitti.astype({'whole_name': np.int64})
    count = 0
    total = sum([result['pred_instances_3d']['scores_3d'].shape[0] for result in results])
    
    print(len(results))
    for j, result in enumerate(results):
        lidar_boxes = result['pred_instances_3d']['bboxes_3d'].tensor
        scores = result['pred_instances_3d']['scores_3d']
        labels = result['pred_instances_3d']['labels_3d']
        if not argo:
            _w2k = waymo2kitti[waymo2kitti['whole_name'] == result['sample_idx']]
            if _w2k.shape[0] == 0:
                continue
            timestamp = _w2k['timestamp'].values.item()
            log_id = _w2k['log_id'].values.item()
        else:
            _a2k = argo_idx[argo_idx['idx'] == result['sample_idx']]
            log_id = _a2k['log_id'].item()
            timestamp = _a2k['timestamp'].item()

        for i in range(lidar_boxes.shape[0]):
            if len(df.index) % 100 == 0:
                print(len(df.index), total)
            data_row = parse_one_object(i, lidar_boxes, scores, labels, timestamp, log_id)
            df.loc[len(df.index)] = data_row
            if df.shape[0] % 50000 == 0 and j != 0:
                print('Writing to ', os.path.join(save_path, 'intermediate', f'{count}.feather'))
                df = df.astype(column_dtypes_dets_wo_traj)
                feather.write_feather(df, os.path.join(save_path, 'intermediate', f'{count}.feather'))
                count += 1
                df = pd.DataFrame(columns=column_names)
    df = df.astype(column_dtypes_dets_wo_traj)
    feather.write_feather(df, os.path.join(save_path, 'intermediate', f'{count}.feather')) # save_path, 'intermediate', 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GenerateAV2DetFile')
    parser.add_argument('file_path')
    parser.add_argument('detection_type')
    parser.add_argument('percentage')
    parser.add_argument('--argo', default=False)
    args = parser.parse_args()

    save_dir = '/workspace/ExchangeWorkspace/detections_from_pp_sv2_format'
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_0.9_0.9/0.9_train_detector.bin'] 
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_PC_SIZES_0.9_0.1/0.1_val_detector.bin']
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_PC_REST_0.9_0.1/0.1_val_detector.bin']
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_TRACK_ALL_0.1_0.1/0.1_val_detector.bin']
    main(args.file_path, save_dir, args.detection_type, args.percentage, args.argo)

