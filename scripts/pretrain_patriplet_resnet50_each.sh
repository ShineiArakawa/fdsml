#!/bin/bash

# License
# -------
# Copyright (c) 2026 Shinei Arakawa
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=48

NUM_GPUS=1
DSET_PATH=datasets/finefractaldb/images                                     # 1M training images
LABEL_CSV_DIR=datasets/finefractaldb/params
INSTANCES_PER_CLASS=8                                                       # 8 instances in one mini-batch
CLASSES_IN_BATCH=32                                                         # 32 classes in one mini-batch
NUM_ACCUMULATIONS=1                                                         # No gradient accumulation, resulting in effective batch size of 256 batch_gpu x 1 rounds x 1 gpus = 256
BATCH_GPU=$((INSTANCES_PER_CLASS * CLASSES_IN_BATCH))                       # ~ 256             images
TOTAL_NUM_IMGS=$(( ${BATCH_GPU} << 19 ))                                    # ~ 256 x   524,288 images = 134,217,728 images ~     1M imgs x 134.22 epochs
STATUS_NUM_IMGS=$(( ${BATCH_GPU} << 10 ))                                   # ~ 256 x     1,024 images =     262,144 images ~  0.26M imgs
CHECKPOINT_NUM_IMGS=$(( ${BATCH_GPU} << 14 ))                               # ~ 256 x    16,384 images =   4,194,304 images ~  4.19M imgs
SNAPSHOT_NUM_IMGS=$(( ${BATCH_GPU} << 15 ))                                 # ~ 256 x    32,768 images =   8,388,608 images ~  8.39M imgs

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    pretrain.py \
    --run_dir_root=out/fdsml \
    --seed=0 \
    --net=resnet50 \
    --projection_dim=512 \
    --compile=max-autotune \
    --miner=each_class \
    --dset=${DSET_PATH} \
    --param_dir=${LABEL_CSV_DIR} \
    --insts_per_class=${INSTANCES_PER_CLASS} \
    --batch_gpu=${BATCH_GPU} \
    --batch_size=$((NUM_GPUS * BATCH_GPU * NUM_ACCUMULATIONS)) \
    --lr=0.01 \
    --total_nimg=${TOTAL_NUM_IMGS} \
    --status_nimg=${STATUS_NUM_IMGS} \
    --checkpoint_nimg=${CHECKPOINT_NUM_IMGS} \
    --snapshot_nimg=${SNAPSHOT_NUM_IMGS}
