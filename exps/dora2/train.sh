#!/bin/bash
set -e  # 遇到错误立即停止脚本

# =========================================================
# 1. 获取脚本所在的绝对路径
# =========================================================
# 无论你在哪个目录运行 ./train.sh，BASE_DIR 都会被解析为 train.sh 所在的文件夹路径
BASE_DIR=$(cd "$(dirname "$0")" && pwd)

echo "当前脚本所在目录: $BASE_DIR"

# =========================================================
# 2. 用户配置区 (所有文件路径都基于 BASE_DIR)
# =========================================================
TOTAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=1

# Python 脚本绝对路径
SCRIPT_PATH="$BASE_DIR/train.py"

# DeepSpeed 配置文件绝对路径
DS_CONFIG="$BASE_DIR/ds_config.json"

# 虚拟环境路径
VENV_PATH="/dfs/data/uv-venv/huggingface/bin/activate"
# ---------------------------------------------------------

# =========================================================
# 3. 环境检查与激活
# =========================================================
# 检查 Python 脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到训练脚本: $SCRIPT_PATH"
    exit 1
fi

# 检查 DeepSpeed 配置文件是否存在
if [ ! -f "$DS_CONFIG" ]; then
    echo "错误: 找不到 DeepSpeed 配置文件: $DS_CONFIG"
    exit 1
fi

# 激活虚拟环境
if [ -f "$VENV_PATH" ]; then
    echo "激活虚拟环境: $VENV_PATH"
    source "$VENV_PATH"
else
    echo "警告: 找不到虚拟环境路径 $VENV_PATH"
    echo "尝试使用当前系统 Python..."
fi

# =========================================================
# 4. 资源检测与启动
# =========================================================
# 自动获取当前机器的 GPU 数量
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "---------------------------------------------------"
echo "检测到 GPU 数量 : $NUM_GPUS"
echo "目标总 Batch Size : $TOTAL_BATCH_SIZE"
echo "单卡 Batch Size   : $PER_DEVICE_BATCH_SIZE"
echo "DeepSpeed 配置文件: $DS_CONFIG"
echo "---------------------------------------------------"

# 使用 deepspeed 启动命令
# 注意：我们这里切换到脚本所在目录再运行，或者使用绝对路径
# 为了确保 Python 脚本内部生成的 logs/checkpoints 目录结构清晰，
# 建议 cd 到脚本目录执行，或者确保 Python 脚本里 output_dir 写的是绝对路径。
# 这里采用 cd 进去执行的方式，最为稳妥。

cd "$BASE_DIR"

deepspeed --num_gpus $NUM_GPUS "$SCRIPT_PATH" \
    --deepspeed "$DS_CONFIG" \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE