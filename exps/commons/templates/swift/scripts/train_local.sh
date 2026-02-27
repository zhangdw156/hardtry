#!/bin/bash
# 步骤：ms-swift SFT 训练。可从任意目录执行（仅依赖脚本自身路径，执行前会 cd 到实验目录）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
cd "$EXP_DIR" || exit 1

SFT_CONFIG="/dfs/data/work/hardtry/exps/${EXP_NAME}/configs/sft_config.yaml"

source /dfs/data/uv-venv/modelscope/bin/activate

OMP_NUM_THREADS=4 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    swift sft --config "$SFT_CONFIG"
