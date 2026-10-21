#!/bin/bash

# License
# -------
# Copyright (c) 2026 Shinei Arakawa
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

DSET_DIR=datasets/finefractaldb
COMMAND_PREFIX=""

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi


# Generate 1,000 IFS parameters and save them to a pickle file
${COMMAND_PREFIX} python -m finefractaldb.ifs \
    --save_path ${DSET_DIR}/ifs_params.pkl \
    --num_systems 1000

# Render the fractal images based on the generated parameters
${COMMAND_PREFIX} python -m finefractaldb.render \
    --param_file ${DSET_DIR}/ifs_params.pkl \
    --out_dir ${DSET_DIR} \
    --patch

# Compute mean and std of parameters. These values will be used for normalization during training.
${COMMAND_PREFIX} python -m tools.compute_param_stats \
    --param_dir ${DSET_DIR}/params

# Create a cache file that contains all parameters in a single file for faster loading during training.
${COMMAND_PREFIX} python -m tools.create_param_cache \
    --param_dir ${DSET_DIR}/params
