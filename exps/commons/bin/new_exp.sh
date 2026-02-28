#!/usr/bin/env bash
# 从 templates 复制生成新实验目录，并将占位符 __EXP_NAME__ 替换为实验名。
# 用法: new_exp.sh <verl|swift> <实验名>

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
readonly COMMONS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$COMMONS_ROOT/../.." && pwd)"
readonly TEMPLATES_DIR="$COMMONS_ROOT/templates"

usage() {
    echo "用法: $0 <verl|swift> <实验名>"
    echo "  verl  - GRPO/强化学习（参考 exps/verl6/configs）"
    echo "  swift - ms-swift SFT（参考 exps/full5/configs）"
    echo "示例: $0 verl verl9  或  $0 swift full9"
    exit 1
}

# 原地替换：兼容 GNU sed 与 BSD sed
_sed_replace_all() {
    local dir="$1"
    local from="$2"
    local to="$3"
    if sed --version 2>/dev/null | grep -q GNU; then
        find "$dir" -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.sh" -o -name "*.md" \) -exec sed -i "s/${from}/${to}/g" {} \;
    else
        find "$dir" -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.sh" -o -name "*.md" \) -exec sed -i '' "s/${from}/${to}/g" {} \;
    fi
}

# --- 参数 ---
[[ -n "${1:-}" && -n "${2:-}" ]] || usage
readonly TEMPLATE_TYPE="$1"
readonly EXP_NAME="$2"
readonly EXP_DIR="$REPO_ROOT/exps/$EXP_NAME"

[[ ! -d "$EXP_DIR" ]] || { echo "错误: 已存在 $EXP_DIR" >&2; exit 1; }

# --- 复制模板 ---
case "$TEMPLATE_TYPE" in
    verl)  cp -R "$TEMPLATES_DIR/verl" "$EXP_DIR" ;;
    swift) cp -R "$TEMPLATES_DIR/swift" "$EXP_DIR" ;;
    *)     echo "错误: 未知类型 $TEMPLATE_TYPE，支持 verl、swift" >&2; exit 1 ;;
esac

# --- 替换占位符 ---
_sed_replace_all "$EXP_DIR" "__EXP_NAME__" "$EXP_NAME"

# 重命名 run___EXP_NAME__.yaml -> run_<实验名>.yaml
placeholder_yaml="$EXP_DIR/configs/run___EXP_NAME__.yaml"
final_yaml="$EXP_DIR/configs/run_${EXP_NAME}.yaml"
[[ -f "$placeholder_yaml" ]] && mv "$placeholder_yaml" "$final_yaml"

# --- 提示 ---
echo "已生成: $EXP_DIR"
case "$TEMPLATE_TYPE" in
    verl)
        echo "请补齐 configs（参考 exps/verl6/configs），再执行: bash exps/$EXP_NAME/run_local.sh"
        ;;
    swift)
        echo "请补齐 configs（参考 exps/full5/configs）并修改 scripts/merge_swift_fsdp_local.sh 中的 CKPT_PATH，再执行: bash exps/$EXP_NAME/run_local.sh"
        ;;
esac
