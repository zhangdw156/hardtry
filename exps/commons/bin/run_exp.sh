#!/usr/bin/env bash
# 通过实验名（及可选入口脚本）启动实验。从仓库根目录执行实验目录下的脚本。
# 用法: run_exp.sh <实验名> [入口脚本]
# 示例: run_exp.sh verl17
#       run_exp.sh verl18 run_local.sh
#       run_exp.sh verl12 run_train_only.sh

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
readonly COMMONS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$COMMONS_ROOT/../.." && pwd)"
readonly EXPS_DIR="$REPO_ROOT/exps"

usage() {
    echo "用法: $0 <实验名> [入口脚本]"
    echo "  实验名   - exps 下实验目录名，如 verl17、verl18"
    echo "  入口脚本 - 可选，默认为 run_local.sh；如 run_train_only.sh"
    echo "示例:"
    echo "  $0 verl17              # 执行 exps/verl17/run_local.sh"
    echo "  $0 verl18 run_local.sh"
    echo "  $0 verl12 run_train_only.sh"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

EXP_NAME="$1"
ENTRY_SCRIPT="${2:-run_local.sh}"
EXP_DIR="$EXPS_DIR/$EXP_NAME"
SCRIPT_PATH="$EXP_DIR/$ENTRY_SCRIPT"

if [ ! -d "$EXP_DIR" ]; then
    echo "错误: 实验目录不存在: $EXP_DIR" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 入口脚本不存在: $SCRIPT_PATH" >&2
    exit 1
fi

if [ ! -x "$SCRIPT_PATH" ]; then
    # 用 bash 执行，不依赖可执行位
    :
fi

cd "$REPO_ROOT" || exit 1
echo ">>> 实验: $EXP_NAME | 入口: $ENTRY_SCRIPT"
echo ">>> 目录: $EXP_DIR"
echo "=========================================="
exec bash "$SCRIPT_PATH"
