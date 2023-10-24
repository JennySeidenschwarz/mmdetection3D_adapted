# Copyright (c) OpenMMLab. All rights reserved.
import torch.distributed as dist
import datetime
import wandb
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend
'''
import sys
import traceback
class TracePrints(object):
  def __init__(self):
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)

sys.stdout = TracePrints()
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('percentage_train', help='1-percentage_train will be the percentage of dataset for training')
    parser.add_argument('percentage_val', help='1-percentage_val will be the percentage of dataset for validation')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--test_detection_set', default='val_detector')
    parser.add_argument('--train_detection_set', default='train_detector')
    parser.add_argument('--pseudo_label_path', default='')
    parser.add_argument('--all_car', default=False)
    parser.add_argument('--stat_as_ignore_region', default=False)
    parser.add_argument('--filter_stat_before', default=False)

    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument(
        '--ann_file2', default='', help='If combining two annotation files')
    parser.add_argument(
            '--args.data_root2', default='')
    parser.add_argument(
            '--args.info_path2', default='')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


from torch import distributed as dist
def get_dist_info():

    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def main():
    from torch import distributed as dist
    rank, world_size = get_dist_info()
    print(world_size, rank)
    '''
    if rank == 0:
        import shutil
        shutil.copyfile('mmengine_to_update/base_dataset.py', '/opt/conda/lib/python3.7/site-packages/mmengine/dataset/base_dataset.py')
        shutil.copyfile('mmengine_to_update/io.py', '/opt/conda/lib/python3.7/site-packages/mmengine/fileio/io.py')
        if os.path.exists('/mmdetection3d/mmdet3d/evaluation/metrics/'):
            shutil.rmtree('/mmdetection3d/mmdet3d/evaluation/metrics/')
            print("deleted", '/mmdetection3d/mmdet3d/evaluation/metrics/')
        shutil.copytree('mmdet3d/evaluation/metrics/', '/mmdetection3d/mmdet3d/evaluation/metrics/')
        shutil.copyfile('mmdet3d/models/task_modules/coders/delta_xyzwhlr_bbox_coder.py', '/mmdetection3d/mmdet3d/models/task_modules/coders/delta_xyzwhlr_bbox_coder.py')
        if os.path.exists('/mmdetection3d/mmdet3d/datasets/'):
            shutil.rmtree('/mmdetection3d/mmdet3d/datasets/')
            print('deleted', '/mmdetection3d/mmdet3d/datasets/')
        shutil.copytree('mmdet3d/datasets/', '/mmdetection3d/mmdet3d/datasets/')
        if os.path.exists('/mmdetection3d/mmdet3d//datasets/transforms/'):
            shutil.rmtree('/mmdetection3d/mmdet3d//datasets/transforms/')
        shutil.copytree('mmdet3d//datasets/transforms/', '/mmdetection3d/mmdet3d//datasets/transforms/')
        shutil.copyfile('mmdet3d/models/task_modules/assigners/max_3d_iou_assigner.py', '/mmdetection3d/mmdet3d/models/task_modules/assigners/max_3d_iou_assigner.py')
    dist.barrier()
    '''
    # dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10000))
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.train_dataloader.dataset['dataset']['percentage'] = float(args.percentage_train)
    if args.ann_file2 is not '':
        cfg.db_sampler['data_root2'] = args.data_root2
        cfg.db_sampler['info_path2'] = args.info_path2
        cfg.train_dataloader.dataset.dataset['dataset']['ann_file2'] = args.ann_file2
    cfg.train_dataloader.dataset.dataset['detection_type'] = args.train_detection_set
    cfg.train_dataloader.dataset.dataset['all_car'] = args.all_car
    cfg.train_dataloader.dataset.dataset['stat_as_ignore_region'] = args.stat_as_ignore_region
    cfg.train_dataloader.dataset.dataset['filter_stat_before'] = args.filter_stat_before

    if args.pseudo_label_path != '':
        cfg.train_dataloader.dataset.dataset['pseudo_labels'] = args.pseudo_label_path
    cfg.val_dataloader.dataset['percentage'] = float(args.percentage_val)
    cfg.test_dataloader.dataset['percentage'] = float(args.percentage_val)
    cfg.test_dataloader.dataset['detection_type'] = args.test_detection_set

    cfg.val_evaluator['percentage'] = float(args.percentage_val)
    cfg.val_evaluator['detection_type'] = cfg.val_dataloader.dataset.detection_type
    cfg.val_evaluator['all_car'] = cfg.val_dataloader.dataset.all_car
    cfg.test_evaluator['percentage'] = float(args.percentage_val)
    cfg.test_evaluator['detection_type'] = cfg.test_dataloader.dataset.detection_type
    cfg.test_evaluator['all_car'] = cfg.test_dataloader.dataset.all_car
    if args.test_detection_set == 'val_evaluation':
        data_root = '/workspace/waymo_kitti_format/kitti_format/'
        rel_annotations_dir = '../../waymo_kitti_format_annotaions/kitti_format'
        cfg.test_dataloader.dataset['ann_file'] = f'{rel_annotations_dir}/waymo_infos_val.pkl'
        cfg.test_evaluator['ann_file'] = f'{data_root}/{rel_annotations_dir}/waymo_infos_val.pkl'

    # add wandb
    wandb.login(key='3b716e6ab76d92ef92724aa37089b074ef19e29c') 
    print('logged into wandb...')
    cfg['log_config'] = {'hooks': [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={
                 'project': 'TrainPointPillars',
                 'name': osp.splitext(osp.basename(args.config))[0]},
             interval=10,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100, 
             bbox_score_thr=0.3)]}
    
    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0] + f'_{args.percentage_train}_{args.percentage_val}')
    
    cfg.val_evaluator['work_dir'] = cfg.work_dir
    cfg.test_evaluator['work_dir'] = cfg.work_dir 
    
    if args.test:
        test(cfg, args)
    else:
        train(cfg, args)


def test(cfg, args):
    cfg.train_dataloader = None
    cfg.train_cfg = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None
    cfg.val_dataloader = None
    cfg.val_cfg = None
    cfg.val_evaluator = None

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()


def train(cfg, args):
    cfg.val_dataloader = None
    cfg.val_cfg = None
    cfg.val_evaluator = None
    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume
    
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    print('Starting training...')
    # start training
    runner.train()


if __name__ == '__main__':
    main()
