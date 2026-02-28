#!/usr/bin/env bash
# 对已有实验目录应用 exps/commons/configs/global.yaml，将 __WORK_ROOT__ 等占位符替换为本机路径。
# 用法: apply_global_config.sh <实验目录>
# 适用于非 new_exp 生成的实验（如 baseline_8b），或 global.yaml 变更后希望同步的实验。

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
readonly COMMONS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly GLOBAL_CONFIG="$COMMONS_ROOT/configs/global.yaml"

usage() {
    echo "用法: $0 <实验目录>"
    echo "示例: $0 exps/baseline_8b"
    exit 1
}

_sed_replace_all() {
    local dir="$1"
    local from="$2"
    local to="$3"
    if sed --version 2>/dev/null | grep -q GNU; then
        find "$dir" -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.sh" -o -name "*.md" \) -exec sed -i "s|${from}|${to}|g" {} \;
    else
        find "$dir" -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.sh" -o -name "*.md" \) -exec sed -i '' "s|${from}|${to}|g" {} \;
    fi
}

_apply_global_config() {
    local dir="$1"
    local config_path="$2"
    [[ -f "$config_path" ]] || return 0
    local key val ph
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        if [[ "$line" =~ ^([a-z_]+):[[:space:]]*(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]}"
            val="${val%%#*}"
            val="$(printf '%s' "$val" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            [[ "$key" == "train_n_gpus" || "$key" == "eval_tensor_parallel_size" ]] && continue
            ph="__$(echo "$key" | tr 'a-z' 'A-Z')__"
            _sed_replace_all "$dir" "$ph" "$val"
        fi
    done < "$config_path"
}

[[ -n "${1:-}" ]] || usage
EXP_DIR="$(cd "$1" && pwd)"
[[ -f "$GLOBAL_CONFIG" ]] || {
    echo "错误: 未找到 $GLOBAL_CONFIG" >&2
    exit 1
}

_apply_global_config "$EXP_DIR" "$GLOBAL_CONFIG"
echo "已对 $EXP_DIR 应用 exps/commons/configs/global.yaml"
