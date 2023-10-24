import argparse
# from __future__ import print_function
from pyarrow import feather
import os
from waymo_open_dataset.protos import metrics_pb2
import tensorflow as tf
from collections import defaultdict
from scipy.spatial.transform import Rotation
import pandas as pd


'''
WAYMO CLASSES:
1 = VEHICLE
2 = PEDESTRIAN
3 = SIGN
4 = CYCLIST
'''

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


def main(file, save_dir):
    detections = metrics_pb2.Objects()
    with open(file, 'rb') as f:
        detections.ParseFromString(f.read())
    
    detections_df = extract_labels(detections)
    save_path = os.path.join(save_dir, file.split('/')[1], file.split('/')[2][:-3] + 'feather')

    os.makedirs(os.path.join(save_dir, file.split('/')[1]), exist_ok=True)
    feather.write_feather(detections_df, save_path)
        
    print(f"Stored to detections converted from {file} to {save_path}")


def extract_labels(detections):
    df = pd.DataFrame(columns=column_names)
    for i, obj in enumerate(detections.objects):
        if i % 100 == 0:
            print(i, len(detections.objects))
        r = Rotation.from_euler('z', obj.object.box.heading)
        quat = r.as_quat()
        data_row = [
            obj.context_name.split('_')[0],
            obj.frame_timestamp_micros,
            -1,
            'TYPE_VECHICLE',
            obj.object.box.length,
            obj.object.box.width,
            obj.object.box.height,
            quat[3],
            quat[0],
            quat[1],
            quat[2],
            obj.object.box.heading,
            obj.object.box.center_x,
            obj.object.box.center_y,
            obj.object.box.center_z,
            -1,
            obj.score]
        df.loc[len(df.index)] = data_row

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GenerateAV2DetFile')
    parser.add_argument('file_path')
    args = parser.parse_args()

    save_dir = '/workspace/ExchangeWorkspace/detections_from_pp_sv2_format'
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_0.9_0.9/0.9_train_detector.bin'] 
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_PC_SIZES_0.9_0.1/0.1_val_detector.bin']
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_REG_ALL_PC_REST_0.9_0.1/0.1_val_detector.bin']
    # file_path = ['work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car_GNN_90_feather_TRACK_ALL_0.1_0.1/0.1_val_detector.bin']
    file_path = args.file_path
    main(file_path, save_dir)

'''
object {
  box {
    center_x: 19.480070895238896
    center_y: -50.34639710054398
    center_z: 2.299644261646847
    width: 2.7734957353748575
    length: 6.179721445654854
    height: 2.6500000000000057
    heading: 3.1052620840875065
  }
  metadata {
    speed_x: -5.067136851525712e-28
    speed_y: -5.067136851525712e-28
    accel_x: 5.630151460658475e-28
    accel_y: 5.630151460658475e-28
  }
  type: TYPE_VEHICLE
  id: "3eaSUD7ej2F7gc9Bisx5VQ"
  num_lidar_points_in_box: 146
}
score: 0.5
context_name: "11048712972908676520_545_000_565_000"
frame_timestamp_micros: 1522684691238230
'''

