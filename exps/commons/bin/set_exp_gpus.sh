#!/usr/bin/env bash
# 将指定实验目录下的训练/评估用卡数写入对应配置与脚本。
# Verl：verl_train_config、verl_common_config*.yaml（训练）、vllm_config*.yaml（评估）。
# Swift：scripts/train_local.sh 中的 NPROC_PER_NODE、CUDA_VISIBLE_DEVICES（训练），vllm_config*.yaml（评估）。
# 用法: set_exp_gpus.sh <实验目录> [训练用卡数] [评估用卡数]
# 省略后两参时从 exps/commons/configs/global.yaml 读取。

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
readonly COMMONS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly GLOBAL_CONFIG="$COMMONS_ROOT/configs/global.yaml"

usage() {
    echo "用法: $0 <实验目录> [训练用卡数] [评估用卡数]"
    echo "示例: $0 exps/verl7    # 使用默认"
    echo "      $0 exps/verl7 4 4"
    exit 1
}

# 原地 sed：兼容 GNU sed 与 BSD sed
_sed_inplace() {
    local file="$1"
    shift
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "$@" "$file"
    else
        sed -i '' "$@" "$file"
    fi
}

# 更新某子目录下的 verl 相关 YAML
_update_verl_in() {
    local subdir="$1"
    local dir="$EXP_DIR/$subdir"
    [[ -d "$dir" ]] || return 0
    local f
    for f in verl_train_config.yaml verl_common_config.yaml verl_common_config_egpo.yaml; do
        [[ -f "$dir/$f" ]] || continue
        _sed_inplace "$dir/$f" -E "s/^(  num_workers: )[0-9]+([^0-9].*)?$/\1${TRAIN_GPUS}\2/"
        _sed_inplace "$dir/$f" -E "s/^(  n_gpus_per_node: )[0-9]+([^0-9].*)?$/\1${TRAIN_GPUS}\2/"
        _sed_inplace "$dir/$f" -E "s/^(  nnodes: )[0-9]+([^0-9].*)?$/\11\2/"
    done
}

# 更新某子目录下的 vLLM YAML
_update_vllm_in() {
    local subdir="$1"
    local dir="$EXP_DIR/$subdir"
    [[ -d "$dir" ]] || return 0
    local f
    for f in vllm_config.yaml vllm_config4.yaml; do
        [[ -f "$dir/$f" ]] || continue
        _sed_inplace "$dir/$f" -E "s/^([[:space:]]*tensor_parallel_size: )[0-9]+([^0-9].*)?$/\1${EVAL_GPUS}\2/"
    done
}

# 更新 Swift 实验 scripts/train_local.sh 中的 NPROC_PER_NODE、CUDA_VISIBLE_DEVICES
_update_swift_train_sh() {
    local sh_path="$EXP_DIR/scripts/train_local.sh"
    [[ -f "$sh_path" ]] || return 0
    grep -q 'NPROC_PER_NODE=' "$sh_path" || return 0
    grep -q 'CUDA_VISIBLE_DEVICES=' "$sh_path" || return 0
    local cuda_devs
    if [[ "$TRAIN_GPUS" -gt 0 ]]; then
        cuda_devs=$(seq -s, 0 $((TRAIN_GPUS - 1)))
    else
        cuda_devs="0"
    fi
    _sed_inplace "$sh_path" -E "s/^(NPROC_PER_NODE=)[0-9]+/\1${TRAIN_GPUS}/"
    _sed_inplace "$sh_path" -E "s/^(CUDA_VISIBLE_DEVICES=)[0-9,]+/\1${cuda_devs}/"
}

# --- 参数与默认值 ---
[[ -n "${1:-}" ]] || usage
EXP_DIR="$(cd "$1" && pwd)"
TRAIN_GPUS="${2:-$(grep -E '^train_n_gpus:' "$GLOBAL_CONFIG" 2>/dev/null | sed -E 's/^train_n_gpus:[[:space:]]*//' || echo "2")}"
EVAL_GPUS="${3:-$(grep -E '^eval_tensor_parallel_size:' "$GLOBAL_CONFIG" 2>/dev/null | sed -E 's/^eval_tensor_parallel_size:[[:space:]]*//' || echo "4")}"

# --- 执行 ---
for sub in configs conf; do
    _update_verl_in "$sub"
    _update_vllm_in "$sub"
done
_update_swift_train_sh

echo "实验目录: $EXP_DIR"
echo "训练用卡: $TRAIN_GPUS  评估用卡(tensor_parallel_size): $EVAL_GPUS"
echo "已更新：verl 训练 YAML、vllm 评估 YAML；若存在 scripts/train_local.sh（含 NPROC_PER_NODE/CUDA_VISIBLE_DEVICES）则已按训练用卡数更新。"
