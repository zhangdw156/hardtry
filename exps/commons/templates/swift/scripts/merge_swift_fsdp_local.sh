#!/bin/bash
# 步骤：自动合并「最近一次」Swift/FSDP checkpoint 为完整模型（与 verl 的 merge_verl_fsdp_local 行为一致）。
# BASE_MODEL_PATH 与训练一致：从 configs/sft_config.yaml 的 model 读取，避免 Thinking/Instruct 不一致。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 按环境修改：merge 用虚拟环境（如 huggingface）
source __VENV_MERGE__/bin/activate

# 与 sft_config / vllm_config 一致；自动找最近 run 与 checkpoint
CHECKPOINT_BASE="__WORK_ROOT__/checkpoints/__EXP_NAME__"
TARGET_DIR="__MODELS_ROOT__/hardtry-4b-__EXP_NAME__"

# 训练基座与 merge 必须一致：从 sft_config 读取，fallback 为默认 Instruct
SFT_CONFIG="$EXP_DIR/configs/sft_config.yaml"
BASE_MODEL_PATH="$(grep -E '^model:' "$SFT_CONFIG" 2>/dev/null | head -1 | sed -E 's/^model:[[:space:]]*["]?//;s/["]?[[:space:]]*$//')"
if [[ -z "$BASE_MODEL_PATH" ]]; then
    echo "警告: 未从 $SFT_CONFIG 读取到 model，使用默认 Qwen3-4B-Instruct-2507" >&2
    BASE_MODEL_PATH="__MODELS_ROOT__/Qwen3-4B-Instruct-2507"
fi

bash "$REPO_ROOT/exps/commons/sbin/merge_swift_fsdp_auto.sh" \
    "$CHECKPOINT_BASE" \
    "$TARGET_DIR" \
    "$BASE_MODEL_PATH"
