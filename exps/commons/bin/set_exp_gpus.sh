#!/bin/bash
# 统一设置某实验的「训练用卡数」与「评估用卡数」到该实验目录下的 verl / vLLM 配置中。
# 用法:
#   bash exps/commons/bin/set_exp_gpus.sh <实验目录> [训练用卡数] [评估用卡数]
# 若省略后两个参数，从 exps/commons/configs/default_exp_resources.yaml 读取默认值。
# 适用范围：verl 实验（改 verl_common_config*.yaml + vllm_config*.yaml）；swift/full/lora 等仅改 vllm 评估用卡。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMONS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RESOURCES="$COMMONS_ROOT/configs/default_exp_resources.yaml"

usage() {
    echo "用法: $0 <实验目录> [训练用卡数] [评估用卡数]"
    echo "  例: $0 exps/verl7"
    echo "  例: $0 exps/verl7 4 4"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi
EXP_DIR="$(cd "$1" && pwd)"

# 解析或读取默认的 训练用卡 / 评估用卡
if [ -n "$2" ]; then
    TRAIN_GPUS="$2"
else
    if [ -f "$DEFAULT_RESOURCES" ]; then
        TRAIN_GPUS=$(grep -E '^train_n_gpus:' "$DEFAULT_RESOURCES" | sed -E 's/^train_n_gpus:[[:space:]]*//')
    fi
    TRAIN_GPUS="${TRAIN_GPUS:-2}"
fi

if [ -n "$3" ]; then
    EVAL_GPUS="$3"
else
    if [ -f "$DEFAULT_RESOURCES" ]; then
        EVAL_GPUS=$(grep -E '^eval_tensor_parallel_size:' "$DEFAULT_RESOURCES" | sed -E 's/^eval_tensor_parallel_size:[[:space:]]*//')
    fi
    EVAL_GPUS="${EVAL_GPUS:-4}"
fi

# 兼容 macOS 与 Linux 的 sed -i
sed_inplace() {
    local f="$1"
    shift
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "$@" "$f"
    else
        sed -i '' "$@" "$f"
    fi
}

# 在给定目录下更新 verl 训练用卡（num_workers, n_gpus_per_node）；nnodes 固定为 1
update_verl_in_dir() {
    local subdir="$1"
    local dir="$EXP_DIR/$subdir"
    [ -d "$dir" ] || return 0
    for f in verl_common_config.yaml verl_common_config_egpo.yaml; do
        [ -f "$dir/$f" ] || continue
        sed_inplace "$dir/$f" -E "s/^(  num_workers: )[0-9]+([^0-9].*)?$/\1${TRAIN_GPUS}\2/"
        sed_inplace "$dir/$f" -E "s/^(  n_gpus_per_node: )[0-9]+([^0-9].*)?$/\1${TRAIN_GPUS}\2/"
        sed_inplace "$dir/$f" -E "s/^(  nnodes: )[0-9]+([^0-9].*)?$/\11\2/"
    done
}

# 在给定目录下更新 vLLM 评估用卡（tensor_parallel_size）
update_vllm_in_dir() {
    local subdir="$1"
    local dir="$EXP_DIR/$subdir"
    [ -d "$dir" ] || return 0
    for f in vllm_config4.yaml vllm_config.yaml; do
        [ -f "$dir/$f" ] || continue
        sed_inplace "$dir/$f" -E "s/^([[:space:]]*tensor_parallel_size: )[0-9]+([^0-9].*)?$/\1${EVAL_GPUS}\2/"
    done
}

echo "实验目录: $EXP_DIR"
echo "训练用卡: $TRAIN_GPUS  评估用卡(tensor_parallel_size): $EVAL_GPUS"
echo ""

for sub in configs conf; do
    update_verl_in_dir "$sub"
    update_vllm_in_dir "$sub"
done

echo "已按上述数值更新该实验下 configs/ 与 conf/ 中的 verl_common_config*.yaml 与 vllm_config*.yaml。"
