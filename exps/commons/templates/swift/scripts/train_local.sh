#!/bin/bash
# ms-swift SFT 训练。实验目录为脚本所在目录的上级，config 使用该实验的 configs/sft_config.yaml。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"

# 使用绝对路径，与 full5 等现有实验一致（便于在不同机器/工作目录下执行）
SFT_CONFIG="/dfs/data/work/hardtry/exps/${EXP_NAME}/configs/sft_config.yaml"

source /dfs/data/uv-venv/modelscope/bin/activate

OMP_NUM_THREADS=4 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    swift sft --config "$SFT_CONFIG"
