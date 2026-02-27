#!/bin/bash
# 委托给 commons 的通用评估脚本，使用本实验的 configs。实验目录为脚本所在目录的上级。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"

exec bash "$REPO_ROOT/exps/commons/bin/eval_local.sh" "$EXP_DIR"
