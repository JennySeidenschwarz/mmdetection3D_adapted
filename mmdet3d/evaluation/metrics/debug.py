import torch
import tempfile
import os.path as osp


def format_results_debug():
    eval_tmp_dir = tempfile.TemporaryDirectory()
    pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
    waymo_save_tmp_dir = tempfile.TemporaryDirectory()
    waymo_results_save_dir = waymo_save_tmp_dir.name
    waymo_results_final_path = f'{pklfile_prefix}.bin'

    waymo_root = '/workspace/waymo/waymo_format' #self.data_root
    waymo_tfrecords_dir = osp.join(waymo_root, 'training')
    prefix = '0'
    final_results = torch.load('debug_res.pth')

    from mmdet3d.evaluation.functional.waymo_utils.prediction_to_waymo import \
            Prediction2Waymo
    converter = Prediction2Waymo(
            final_results,
            waymo_tfrecords_dir,
            waymo_results_save_dir,
            waymo_results_final_path,
            prefix,
            {0: 'Car'},
            backend_args=None,
            from_kitti_format=False,
            idx2metainfo=None,
            detection_type='val_detector',
            percentage=0.1)
    converter.convert()
    waymo_save_tmp_dir.cleanup()

    return final_results, waymo_save_tmp_dir


if __name__ == "__main__":
    format_results_debug()
