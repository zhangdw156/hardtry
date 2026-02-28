#!/bin/bash
# 步骤：自动合并「最近一次」Swift/FSDP checkpoint 为完整模型（与 verl 的 merge_verl_fsdp_local 行为一致）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 按环境修改：merge 用虚拟环境（如 huggingface）
source /dfs/data/uv-venv/huggingface/bin/activate

# 与 sft_config / vllm_config 一致，使用相同占位符；自动找最近 run 与 checkpoint
CHECKPOINT_BASE="/dfs/data/work/hardtry/checkpoints/full9"
TARGET_DIR="/dfs/data/models/hardtry-4b-full9"
BASE_MODEL_PATH="/dfs/data/models/Qwen3-4B-Instruct-2507"

bash "$REPO_ROOT/exps/commons/sbin/merge_swift_fsdp_auto.sh" \
    "$CHECKPOINT_BASE" \
    "$TARGET_DIR" \
    "$BASE_MODEL_PATH"
