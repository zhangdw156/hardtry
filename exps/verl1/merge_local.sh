#!/bin/bash

/dfs/data/uv-venv/verl/bin/python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /dfs/data/work/hardtry/checkpoints/verl1/global_step_221/actor \
    --target_dir /dfs/data/models/hardtry-4b-verl1 \
    --use_cpu_initialization