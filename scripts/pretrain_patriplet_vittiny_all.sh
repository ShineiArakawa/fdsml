#!/bin/bash

# License
# -------
# Copyright (c) 2026 Shinei Arakawa
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=48

NUM_GPUS=2
DSET_PATH=datasets/finefractaldb/images                                     # 1M training images
LABEL_CSV_DIR=datasets/finefractaldb/params
INSTANCES_PER_CLASS=16                                                      # 16 instances in one mini-batch
CLASSES_IN_BATCH=16                                                         # 16 classes in one mini-batch
NUM_ACCUMULATIONS=8                                                         # 8 gradient accumulation rounds, resulting in effective batch size of 256 batch_gpu x 8 rounds x 2 gpus = 4,096
BATCH_GPU=$((INSTANCES_PER_CLASS * CLASSES_IN_BATCH))                       # ~ 256             images
TOTAL_NUM_IMGS=$(( ${BATCH_GPU} << 21 ))                                    # ~ 256 x 2,097,152 images = 536,870,912 images ~     1M imgs x 536.87 epochs
STATUS_NUM_IMGS=$(( ${BATCH_GPU} << 10 ))                                   # ~ 256 x     1,024 images =     262,144 images ~  0.26M imgs
CHECKPOINT_NUM_IMGS=$(( ${BATCH_GPU} << 14 ))                               # ~ 256 x    16,384 images =   4,194,304 images ~   4.2M imgs
SNAPSHOT_NUM_IMGS=$(( ${BATCH_GPU} << 17 ))                                 # ~ 256 x   131,072 images =  33,554,432 images ~  33.6M imgs

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    pretrain.py \
    --run_dir_root=out/fdsml \
    --seed=0 \
    --net=deit_tiny_patch16_224_mocov3 \
    --projection_dim=128 \
    --freeze_patch_proj \
    --miner=all \
    --dset=${DSET_PATH} \
    --param_dir=${LABEL_CSV_DIR} \
    --insts_per_class=${INSTANCES_PER_CLASS} \
    --batch_gpu=${BATCH_GPU} \
    --batch_size=$((NUM_GPUS * BATCH_GPU * NUM_ACCUMULATIONS)) \
    --lr=1e-4 \
    --total_nimg=${TOTAL_NUM_IMGS} \
    --status_nimg=${STATUS_NUM_IMGS} \
    --checkpoint_nimg=${CHECKPOINT_NUM_IMGS} \
    --snapshot_nimg=${SNAPSHOT_NUM_IMGS}
