#!/usr/bin/env bash
# 将 checkpoint 根目录下最后一次 global_step_* 的 actor 合并到目标目录。
# 用法: merge_verl_fsdp_auto.sh <CHECKPOINT_BASE> <TARGET_DIR>（仅由实验 scripts 转调）
# 可选环境变量: PYTHON_CMD (默认 /dfs/data/uv-venv/verl/bin/python3)

set -euo pipefail

readonly PYTHON_CMD="${PYTHON_CMD:-/dfs/data/uv-venv/verl/bin/python3}"

usage() {
    echo "用法: $0 <CHECKPOINT_BASE> <TARGET_DIR>"
    echo "示例: $0 /path/to/checkpoints/verl8 /path/to/models/hardtry-4b-verl8"
    exit 1
}

# 在 CHECKPOINT_BASE 下找最大的 global_step_<N> 的 N
find_last_global_step() {
    local base="$1"
    local last=""
    local d
    for d in "$base"/global_step_*; do
        [[ -d "$d" ]] || continue
        local step="${d##*global_step_}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if [[ -z "$last" || "$step" -gt "$last" ]]; then
            last="$step"
        fi
    done
    echo "$last"
}

# --- 参数 ---
CHECKPOINT_BASE="${1:-}"
TARGET_DIR="${2:-}"
[[ -n "$CHECKPOINT_BASE" && -n "$TARGET_DIR" ]] || usage
CHECKPOINT_BASE="$(cd "$CHECKPOINT_BASE" && pwd)"

LAST_STEP="$(find_last_global_step "$CHECKPOINT_BASE")"
[[ -n "$LAST_STEP" ]] || { echo "错误: 在 $CHECKPOINT_BASE 下未找到 global_step_*" >&2; exit 1; }

LOCAL_DIR="$CHECKPOINT_BASE/global_step_${LAST_STEP}/actor"
[[ -d "$LOCAL_DIR" ]] || { echo "错误: 不存在 $LOCAL_DIR" >&2; exit 1; }

# --- 执行 ---
echo "Checkpoint: $LOCAL_DIR"
echo "目标目录:   $TARGET_DIR"
"$PYTHON_CMD" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR" \
    --use_cpu_initialization
