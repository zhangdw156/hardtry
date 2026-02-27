#!/bin/bash
# 委托给 commons 的稳定工具，使用本实验的 configs。实验目录为脚本所在目录的上级。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# scripts 在 exps/<实验名>/scripts/，上三级才是仓库根
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

exec bash "$REPO_ROOT/exps/commons/bin/eval_local.sh" "$EXP_DIR"
