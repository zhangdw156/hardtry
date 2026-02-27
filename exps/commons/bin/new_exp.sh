#!/bin/bash
# 从模板复制生成新实验目录，并替换占位符 __EXP_NAME__。
# 用法: bash exps/commons/bin/new_exp.sh <verl|swift> <实验名>
# 示例: bash exps/commons/bin/new_exp.sh verl verl7
#       bash exps/commons/bin/new_exp.sh swift full8
# 复制后请到 exps/<实验名>/configs 补齐配置（verl 参考 exps/verl6/configs，swift 参考 exps/full5/configs）。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMONS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COMMONS_DIR/../.." && pwd)"
TEMPLATES_DIR="$COMMONS_DIR/templates"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "用法: $0 <verl|swift> <实验名>"
    echo "  verl  - GRPO/强化学习实验（参考 exps/verl6）"
    echo "  swift - ms-swift SFT 实验（参考 exps/full5）"
    echo "示例: $0 verl verl7  或  $0 swift full8"
    exit 1
fi
TEMPLATE_TYPE="$1"
EXP_NAME="$2"
EXP_DIR="$REPO_ROOT/exps/$EXP_NAME"

if [ -d "$EXP_DIR" ]; then
    echo "错误: 目录已存在 $EXP_DIR"
    exit 1
fi

case "$TEMPLATE_TYPE" in
    verl)
        cp -R "$TEMPLATES_DIR/verl" "$EXP_DIR"
        ;;
    swift)
        cp -R "$TEMPLATES_DIR/swift" "$EXP_DIR"
        ;;
    *)
        echo "错误: 未知模板 $TEMPLATE_TYPE，当前支持 verl、swift"
        exit 1
        ;;
esac

# 替换所有文件中的 __EXP_NAME__ 为实验名
if sed --version >/dev/null 2>&1; then
    SED_INPLACE=(sed -i "s/__EXP_NAME__/$EXP_NAME/g")
else
    SED_INPLACE=(sed -i '' "s/__EXP_NAME__/$EXP_NAME/g")
fi
find "$EXP_DIR" -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.sh" -o -name "*.md" \) -exec "${SED_INPLACE[@]}" {} \;

# 模板里配置文件名是 run___EXP_NAME__.yaml，替换后需重命名为 run_<实验名>.yaml
PLACEHOLDER_YAML="$EXP_DIR/configs/run___EXP_NAME__.yaml"
FINAL_YAML="$EXP_DIR/configs/run_${EXP_NAME}.yaml"
if [ -f "$PLACEHOLDER_YAML" ]; then
    mv "$PLACEHOLDER_YAML" "$FINAL_YAML"
fi

case "$TEMPLATE_TYPE" in
    verl)
        echo "已生成实验目录: $EXP_DIR"
        echo "请到 $EXP_DIR/configs 补齐/修改配置（参考 exps/verl6/configs），再执行: bash exps/$EXP_NAME/run_local.sh"
        ;;
    swift)
        echo "已生成实验目录: $EXP_DIR"
        echo "请到 $EXP_DIR/configs 补齐 sft_config.yaml、vllm_config4.yaml、eval_config5.yaml（参考 exps/full5/configs），"
        echo "并修改 scripts/merge_swift_fsdp_local.sh 中的 CKPT_PATH，再执行: bash exps/$EXP_NAME/run_local.sh"
        ;;
esac
