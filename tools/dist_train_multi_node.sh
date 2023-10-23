#!/usr/bin/env bash
  
CONFIG=$1
GPUS=$2
PERCENTAGE_TRAIN=$3
PERCENATGE_VAL=$4
NNODES=${NGC_ARRAY_SIZE}
NODE_RANK=${NGC_ARRAY_INDEX}
PORT=${PORT:-29500}
MASTER_ADDR=${NGC_MASTER_ADDR:-"127.0.0.1"}

printf "${PERCENTAGE_TRAIN}"
printf "${PORT}"
printf "${NNODES}"
printf "${NODE_RANK}"
printf "${MASTER_ADDR}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    ${PERCENTAGE_TRAIN} \
    ${PERCENATGE_VAL} \
    --launcher pytorch ${@:5}
~                                     