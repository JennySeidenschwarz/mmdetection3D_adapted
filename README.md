# Training downstream detector
Directory forked from [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/). It is adapted to load feather files as input for training and evaluation on pointpillars model. The performance is automatically evaluated using the evaluation code from SeMoLi. For training we use the standard hyperparameters for pointpillars training on Waymo Open Dataset. We support training on Waymo Open Dataset and Argoverse2 dataset.

## Installation
If you installed the conda environment from SeMoLi, all libraries are already installed and you can run the code if you activate ```conda activate SeMoLi```. Otherwise, perpare a conda evironment running the following:

```
conda create -n mmdetection3d python=3.9
conda activate mmdetection3d
bash setup.sh
```

## Running the code
In this repository we follow the data split convention of SeMoLi given by:
![data split figure according to SeMoLi](figs/data_splits.pdf)

For training and evaluation set the train and validation label paths by running:
```
export TRAIN_LABELS=<train_label_path>
export VAL_LABELS=<val_label_path>
```

For validation, set the path to the feather file containing ground truth data. If you want to use the ```val_detector``` dataset from SeMoLi for evaluation, set the path to the feather file containing ground truth training data. If you want to use the real validation set, i.e., the ```val_evaluation``` split set the path to the file containing validation set ground truth data. 

For example, if you are using this repository within the SeMoLi repository, the train and real validation set paths are given by:

```
export TRAIN_LABELS=../SeMoLi/data_utils/Waymo_Converted_filtered/train_1_per_frame_remove_non_move_remove_far_filtered_version_city_w0.feather
export VAL_LABELS=../SeMoLi/data_utils/Waymo_Converted_filtered/val_1_per_frame_remove_non_move_remove_far_filtered_version_city_w0.feather
```

The base command for training and evaluation is given by:

```
./tools/dist_train.sh configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_waymo-3d-class_agnostic.py <num_gpus> <percentage_train> <percentage_val> $TRAIN_LABELS $VAL_LABELS --eval --val_detection_set=val_evaluation --auto-scale-lr
```
where
- ```num_gpus``` is the number of GPUs that are used for the training
- ```percentage_train``` is the percentage of training data you want to use for training according to SeMoLi splits
- ```percentage_val``` is 1.0 if you want to use either ```val_detector``` or the real validation set ```val_evaluation```. If you want to use any part of the ```train_detector``` or ```train_gnn``` for evaluation, please set the percentage according to SeMoLi
- ```eval``` if eval is set, you will only evaluate and not train
- ```val_detection``` determines the detection split you want to use, i.e., ```val_detector``` or the real validation set ```val_gnn```
- ```auto-scale-lr``` adapts the learning rate to the batch size according to a given base learning rate



