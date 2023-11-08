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
    
    waymo2kitti = get_waymo2kitti(detection_type, percentage)
    os.makedirs(os.path.join(save_dir, file.split('/')[1]), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'intermediate'))
    
    save_path = os.path.join(save_dir, file.split('/')[1])
    file_name = file.split('/')[2][:-3] + 'feather'
    
    convert(detections, waymo2kitti, save_path, argo=argo)

    paths = glob.glob(f'{save_path}/intermediate/*')
    all_df = None
    for f in paths:
        df = feather.read_feather(f)
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df])
        shutil.rmfile(f)

    feather.write_feather(all_df, os.path.join(save_path))
        
    print(f"Stored to detections converted from {file} to {save_path}")


def parse_one_object(index, lidar_boxes, scores, labels, timestamp, log_id):
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
        class_name,
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
        scores[index]]
    
    return row

def convert(results, waymo2kitti, save_path, argo=False):
    df = pd.DataFrame(columns=column_names)
    for j, result in enumerate(results):
        if j % 100 == 0:
            print(j, len(results))
        lidar_boxes = result['pred_instances_3d']['bboxes_3d'].tensor
        scores = result['pred_instances_3d']['scores_3d']
        labels = result['pred_instances_3d']['labels_3d']
        if not argo:
            _w2k = waymo2kitti['whole_name'] == result['sample_index']
            timestamp = _w2k['timestamp']
            log_ig = _w2k['log_id']
        else:
            log_id = result['sample_index'].split('_')[0]
            timestamp = result['sample_index'].split('_')[1]
        result['sample_index']

        for i in len(result):
            data_row = parse_one_object(i, lidar_boxes, scores, labels, timestamp, log_ig)
            df.loc[len(df.index)] = data_row

            if i * j % 50000 == 0 and j != 0:
                print('Writing to ', os.path.join(save_path, 'intermediate', f'{count}.feather'))
                feather.write_feather(df, os.path.join(save_path, 'intermediate', f'{count}.feather'))
                count += 1
                df = pd.DataFrame(columns=column_names)

        feather.write_feather(df, os.path.join(save_path, 'intermediate', f'{count}.feather'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GenerateAV2DetFile')
    parser.add_argument('file_path')
    parser.add_argument('detection_type')
    parser.add_argument('percentage')
    parser.add_argument('argo')
    args = parser.parse_args()

    save_dir = '/workspace/ExchangeWorkspace/detections_from_pp_sv2_format'
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_0.9_0.9/0.9_train_detector.bin'] 
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_PC_SIZES_0.9_0.1/0.1_val_detector.bin']
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_PC_REST_0.9_0.1/0.1_val_detector.bin']
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_TRACK_ALL_0.1_0.1/0.1_val_detector.bin']

    main(args.file_path, save_dir, args.detection_type, args.percentage, args.argo)

