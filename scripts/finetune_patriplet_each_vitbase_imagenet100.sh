#!/bin/bash

# License
# -------
# Copyright (c) 2026 Shinei Arakawa
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=48

NUM_GPUS=2
DSET_PATH=datasets/imagenet100                                     # 128,500 training images, 5,000 validation images
CKPT_PATH=/path/to/checkpoint.pt                                   # TODO: Set the path to the FDSML pre-trained checkpoint (e.g., out/fdsml/00000-fdsml-*/checkpoints/checkpoint-*kimg-*ep.pt)
NET="deit_base_patch16_224"
BATCH_GPU=512
NUM_ACCUMULATIONS=1
BATCH_SIZE=$((NUM_GPUS * BATCH_GPU * NUM_ACCUMULATIONS))
TOTAL_NUM_IMGS=$(( ${BATCH_SIZE} * 125489 ))
STATUS_NUM_IMGS=$(( ${BATCH_SIZE} << 9 ))
CHECKPOINT_NUM_IMGS=$(( ${BATCH_SIZE} << 14 ))
SNAPSHOT_NUM_IMGS=$(( ${BATCH_SIZE} << 15 ))


if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --standalone \
    finetune.py \
    --run_dir_root="out/fdsml" \
    --seed=0 \
    --net=${NET} \
    --ckpt=${CKPT_PATH} \
    --freeze_patch_proj \
    --dset=${DSET_PATH} \
    --batch_size=$((NUM_GPUS * BATCH_GPU * NUM_ACCUMULATIONS)) \
    --batch_gpu=${BATCH_GPU} \
    --eval_intervals="-799:100,800-:1" \
    --total_nimg=${TOTAL_NUM_IMGS} \
    --status_nimg=${STATUS_NUM_IMGS} \
    --checkpoint_nimg=${CHECKPOINT_NUM_IMGS} \
    --snapshot_nimg=${SNAPSHOT_NUM_IMGS} \
    --precision="bf16"
