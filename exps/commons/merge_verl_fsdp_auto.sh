#!/bin/bash
# 自动取 checkpoint 根目录下「最后一次保存」的 global_step_* 进行 merge，便于一条龙流水线。
# 用法:
#   CHECKPOINT_BASE=/path/to/checkpoints/verl6 TARGET_DIR=/path/to/model bash merge_verl_fsdp_auto.sh
#   bash merge_verl_fsdp_auto.sh /path/to/checkpoints/verl6 /path/to/model
# 可选环境变量: PYTHON_CMD (默认 /dfs/data/uv-venv/verl/bin/python3)

set -e

if [ -n "$2" ]; then
    CHECKPOINT_BASE="$1"
    TARGET_DIR="$2"
fi

if [ -z "$CHECKPOINT_BASE" ] || [ -z "$TARGET_DIR" ]; then
    echo "用法: CHECKPOINT_BASE=... TARGET_DIR=... $0"
    echo "  或: $0 <CHECKPOINT_BASE> <TARGET_DIR>"
    exit 1
fi

PYTHON_CMD="${PYTHON_CMD:-/dfs/data/uv-venv/verl/bin/python3}"

# 查找所有 global_step_* 目录并取步数最大者
LAST_STEP=""
for d in "${CHECKPOINT_BASE}"/global_step_*; do
    [ -d "$d" ] || continue
    step="${d##*global_step_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if [ -z "$LAST_STEP" ] || [ "$step" -gt "$LAST_STEP" ]; then
        LAST_STEP="$step"
    fi
done

if [ -z "$LAST_STEP" ]; then
    echo "错误: 在 ${CHECKPOINT_BASE} 下未找到任何 global_step_* 目录"
    exit 1
fi

LOCAL_DIR="${CHECKPOINT_BASE}/global_step_${LAST_STEP}/actor"
if [ ! -d "$LOCAL_DIR" ]; then
    echo "错误: actor 目录不存在: ${LOCAL_DIR}"
    exit 1
fi

echo "使用最后一次保存的 checkpoint: ${LOCAL_DIR}"
echo "合并目标: ${TARGET_DIR}"
"$PYTHON_CMD" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}" \
    --use_cpu_initialization
