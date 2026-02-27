#!/bin/bash

source /dfs/data/uv-venv/modelscope/bin/activate

OMP_NUM_THREADS=4 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    swift sft --config \
        "/dfs/data/work/hardtry/exps/full6/configs/sft_config.yaml"
