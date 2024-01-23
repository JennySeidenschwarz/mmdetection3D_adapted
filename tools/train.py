# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.datasets.av2_feather_dataset import AV2FeatherDataset
from mmdet3d.datasets.waymo_feather_dataset import WaymoFeatherDataset
from  mmdet3d.evaluation.metrics.av2_metric_feather import AV2MetricFeather
from  mmdet3d.evaluation.metrics.waymo_metric_feather import WaymoMetricFeather
from mmdet3d.datasets.transforms.formating_feather import Pack3DDetInputsFeather
from mmdet3d.datasets.transforms.transforms_3d_feather import PointsRangeFilterFeather, ObjectRangeFilterFeather, GlobalRotScaleTransFeather, RandomFlip3DFeather
from mmdet3d.datasets.transforms.loading_feather import LoadPointsFromFileFeather, LoadAnnotations3DFeather
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    # my added args without default
    parser.add_argument('config', help='train config file path')
    parser.add_argument('percentage_train', help='1-percentage_train will be the percentage of dataset for training')
    parser.add_argument('percentage_val', help='1-percentage_val will be the percentage of dataset for validation')
    parser.add_argument('train_label_path')
    parser.add_argument('val_label_path')

    # my added args with default
    parser.add_argument('--train_label_path_2', default='')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train_detection_set', default='train_detector')
    parser.add_argument('--val_detection_set', default='val_detector')
    parser.add_argument('--test_detection_set', default='')
    parser.add_argument('--test_label_path', default='')
    parser.add_argument('--class_agnostic', default=False)
    parser.add_argument('--filter_stat_before', action='store_false')
    parser.add_argument('--split_path', default='')

    # original args
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
    parser.add_argument('--min_num_pts_filtered', default=0)
    parser.add_argument('--batch_size', type=int, default=2)
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
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    # dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10000))
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.model.test_cfg.pts['score_thr'] = 0.1

    # update train dataloader from parser
    cfg.train_dataloader.dataset['dataset']['percentage'] = float(args.percentage_train)
    cfg.train_dataloader.dataset.dataset['detection_type'] = args.train_detection_set
    cfg.train_dataloader.dataset.dataset['class_agnostic'] = args.class_agnostic
    cfg.train_dataloader.dataset.dataset['filter_stat_before'] = args.filter_stat_before
    cfg.train_dataloader.dataset.dataset['in_channels'] = cfg.model.pts_voxel_encoder.in_channels
    cfg.train_dataloader.batch_size = int(args.batch_size) 
    pseudo_add = f'_{os.path.basename(args.train_label_path)}'[:30]
    cfg.train_dataloader.dataset.dataset['label_path'] = args.train_label_path
    cfg.train_dataloader.dataset.dataset['label_path2'] = args.train_label_path_2
    print('Using Pseudo labels ', cfg.train_dataloader.dataset.dataset['label_path'])

    # update val dataloader and evaluator from parser
    cfg.val_dataloader.dataset['percentage'] = float(args.percentage_val)
    cfg.val_dataloader.dataset['detection_type'] = args.val_detection_set
    cfg.val_dataloader.dataset['in_channels'] = cfg.model.pts_voxel_encoder.in_channels
    cfg.val_dataloader.dataset['filter_stat_before'] = args.filter_stat_before
    cfg.val_evaluator['percentage'] = float(args.percentage_val)
    cfg.val_evaluator['detection_type'] = cfg.val_dataloader.dataset.detection_type
    cfg.val_evaluator['class_agnostic'] = cfg.val_dataloader.dataset.class_agnostic
    cfg.val_dataloader.dataset['label_path'] = args.val_label_path

    # update test dataloader and evaluator from parser
    cfg.test_dataloader.dataset['percentage'] = float(args.percentage_val)
    cfg.test_dataloader.dataset['detection_type'] = args.test_detection_set if args.test_detection_set != '' else args.val_detection_set
    cfg.test_dataloader.dataset['in_channels'] = cfg.model.pts_voxel_encoder.in_channels
    cfg.test_dataloader.dataset['filter_stat_before'] = args.filter_stat_before
    cfg.test_evaluator['percentage'] = float(args.percentage_val)
    cfg.test_evaluator['detection_type'] = cfg.test_dataloader.dataset.detection_type
    cfg.test_evaluator['class_agnostic'] = cfg.test_dataloader.dataset.class_agnostic
    if args.test_label_path != '':
        cfg.test_dataloader.dataset['label_path'] = args.test_label_path
    else:
        cfg.test_dataloader.dataset['label_path'] = args.val_label_path  

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
                                osp.splitext(osp.basename(args.config))[0] + \
                                    f'_{args.percentage_train}_{args.percentage_val}_{args.filter_stat_before}_{args.class_agnostic}_{args.train_detection_set}{pseudo_add}')
    print(cfg.work_dir)
    # set workdir for val and test datasets
    cfg.train_dataloader.dataset.dataset['work_dir'] = cfg.work_dir
    cfg.val_dataloader.dataset['work_dir'] = cfg.work_dir
    cfg.test_dataloader.dataset['work_dir'] = cfg.work_dir
    cfg.val_evaluator['work_dir'] = cfg.work_dir
    cfg.test_evaluator['work_dir'] = cfg.work_dir 
    
    if args.eval:
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

    cfg.load_from = args.resume

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
    
    # start training
    runner.train()


if __name__ == '__main__':
    main()
