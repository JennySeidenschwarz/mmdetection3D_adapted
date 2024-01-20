#!/bin/bash

# mmdetection3d 
python -m pip install numpy==1.22.0
python -m pip install -U openmim
mim install mmcv
mim install mmengine
mim install "mmcv==2.0.0"
mim install "mmdet==3.0.0"
mim install "mmdet3d>=1.1.0"