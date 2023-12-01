# Training downstream detector
Directory forked from https://github.com/open-mmlab/mmdetection3d/

## For training
```
./tools/train_dist.sh configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_waymo-3d-class_agnostic_pseudo.py <num_gpus> <percentage train_pseudo x> <percentage val> --test_detection_set=<detection set val> --train_pseudo_label_path <pseudo_labels_to_use> --auto-scale-lr
```
Example:
```
.tools/train_dist.sh configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_waymo-3d-class_agnostic_pseudo.py 8 0.9 1.0 --test_detection_set=val_detector --train_pseudo_label_path RE_ABLATION_MMMV_P_DP_0.9_0.9_all_egocomp_margin0.6_width25_nooracle_64_3_True_64_3_True_0.5_3.5_0.5_4_3.162277660168379e-06_0.0031622776601683794_16000__NS_MG_32_LN___P___DP___MMMV__P_/train_detector/annotations.feather --auto-scale-lr
```

Default dir for pseudo_labels /workspace/ExchangeWorkspace/detections_train_detector/

## FOr testing
```
./tools/train_dist.sh configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_waymo-3d-class_agnostic_pseudo.py <num_gpus> <percentage train_pseudo x> <percentage val> --test_detection_set=<detection set val> --train_pseudo_label_path <pseudo_labels_to_use> --auto-scale-lr --test --checkpoint <checkpoint> --test_pseudo_label_path <>
```
Default dir for test_pseudo_labels (which is actually true labels) /workspace/ExchangeWorkspace/detections_train_detector/

checkpoints stored in work_dirs


## NGC
ngc command for waymo:
```
ngc batch run --name "DEBUG" --priority NORMAL --order 50 --preempt RUNONCE --min-timeslice 0s --total-runtime 0s --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline "sleep 10h 00m 00s" --result /workspace/result --image "nvidian/dvl/pytorch:mmdetection3d" --org nvidian --team dvl --datasetid 1608966:/workspace/waymo_kitti_format --datasetid 1608876:/workspace/Argoverse2 --datasetid 1609322:/workspace/waymo_kitti_format_annotaions --datasetid 1608604:/workspace/waymo --datasetid 1609977:/workspace/waymo_kitti_format_annotaions_av2kitti --workspace S_tnZ3IETuy3t0iSzV7Oxw:/workspace/mmdetection3d:RW --workspace MTsf2ZBURM6VoTtaCnuskg:/workspace/ExchangeWorkspace:RW --label _ml___pointpillars_INTERSECTION --label _wl___computer_vision --label __TPL_P0 --label __TPL_P1
```

ngc command for av2:
```
```
