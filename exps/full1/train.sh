#!/bin/bash
set -e  # 遇到错误立即停止脚本

# =========================================================
# 1. 获取脚本所在的绝对路径
# =========================================================
BASE_DIR=$(cd "$(dirname "$0")" && pwd)
echo "当前脚本所在目录: $BASE_DIR"

# =========================================================
# 2. 用户配置区
# =========================================================
# Python 脚本绝对路径
SCRIPT_PATH="$BASE_DIR/train.py"

# YAML 配置文件路径
CONFIG_YAML="$BASE_DIR/configs/train_config.yaml"

# 虚拟环境路径
VENV_PATH="/dfs/data/uv-venv/huggingface/bin/activate"

# =========================================================
# 3. 环境检查与激活
# =========================================================
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到训练脚本: $SCRIPT_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_YAML" ]; then
    echo "错误: 找不到 YAML 配置文件: $CONFIG_YAML"
    exit 1
fi

if [ -f "$VENV_PATH" ]; then
    echo "激活虚拟环境: $VENV_PATH"
    source "$VENV_PATH"
else
    echo "警告: 找不到虚拟环境路径 $VENV_PATH"
fi

# =========================================================
# 4. 资源检测与启动
# =========================================================
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "---------------------------------------------------"
echo "检测到 GPU 数量 : $NUM_GPUS"
echo "配置文件        : $CONFIG_YAML"
echo "---------------------------------------------------"

cd "$BASE_DIR"

deepspeed --num_gpus $NUM_GPUS "$SCRIPT_PATH" "$CONFIG_YAML"