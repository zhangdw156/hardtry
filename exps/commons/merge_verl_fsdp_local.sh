#!/bin/bash

##############################
# TODO: To be modified
##############################
LOCAL_DIR="/dfs/data/work/hardtry/checkpoints/verl1/global_step_221/actor"
##############################
# TODO: To be modified
##############################
TARGET_DIR="/dfs/data/models/hardtry-4b-verl1"

/dfs/data/uv-venv/verl/bin/python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${LOCAL_DIR} \
    --target_dir ${TARGET_DIR} \
    --use_cpu_initialization