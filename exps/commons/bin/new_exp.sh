#!/usr/bin/env bash
# 从 templates 复制生成新实验目录；占位符 __EXP_NAME__ 与全局占位符由 configs/global.yaml 替换。
# 用法: new_exp.sh <verl|swift> <实验名>

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
readonly COMMONS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$COMMONS_ROOT/../.." && pwd)"
readonly TEMPLATES_DIR="$COMMONS_ROOT/templates"
readonly GLOBAL_CONFIG="$COMMONS_ROOT/configs/global.yaml"

usage() {
    echo "用法: $0 <verl|swift> <实验名>"
    echo "  verl  - GRPO/强化学习（configs 已内置）"
    echo "  swift - ms-swift SFT（configs 已内置）"
    echo "示例: $0 verl verl9  或  $0 swift full9"
    exit 1
}

# 原地替换：兼容 GNU sed 与 BSD sed（用于不含 / 的替换如 __EXP_NAME__）
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

# 从 configs/global.yaml 读取并替换全局占位符（纯 bash 解析，不依赖 Python yaml）
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

# --- 参数 ---
[[ -n "${1:-}" && -n "${2:-}" ]] || usage
readonly TEMPLATE_TYPE="$1"
readonly EXP_NAME="$2"
readonly EXP_DIR="$REPO_ROOT/exps/$EXP_NAME"

[[ ! -d "$EXP_DIR" ]] || { echo "错误: 已存在 $EXP_DIR" >&2; exit 1; }
[[ -f "$GLOBAL_CONFIG" ]] || {
    echo "错误: 未找到全局配置 $GLOBAL_CONFIG" >&2
    echo "请先按本机环境填写该文件（work_root、models_root、venv_* 等），再运行 new_exp.sh。" >&2
    exit 1
}

# --- 复制模板 ---
case "$TEMPLATE_TYPE" in
    verl)  cp -R "$TEMPLATES_DIR/verl" "$EXP_DIR" ;;
    swift) cp -R "$TEMPLATES_DIR/swift" "$EXP_DIR" ;;
    *)     echo "错误: 未知类型 $TEMPLATE_TYPE，支持 verl、swift" >&2; exit 1 ;;
esac

# --- 替换占位符 ---
_sed_replace_all "$EXP_DIR" "__EXP_NAME__" "$EXP_NAME"
_apply_global_config "$EXP_DIR" "$GLOBAL_CONFIG"

# --- 提示 ---
echo "已生成: $EXP_DIR"
echo "已按 exps/commons/configs/global.yaml 替换实验名与全局路径；执行: bash exps/$EXP_NAME/run_local.sh"
