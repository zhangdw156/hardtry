#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

CONFIG_PATH="$SCRIPT_DIR/configs/sft_config.yaml"

source /dfs/data/uv-venv/modelscope/bin/activate

ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

OMP_NUM_THREADS=$GPU_COUNT \
NPROC_PER_NODE=$GPU_COUNT \
CUDA_VISIBLE_DEVICES=$ALL_GPUS \
    swift sft --config "$CONFIG_PATH"