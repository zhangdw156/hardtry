#!/bin/bash
# 步骤：委托 commons 启动 vLLM 并跑评估。可从任意目录执行（仅依赖脚本自身路径）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# scripts 在 exps/verl6/scripts/，上三级才是仓库根
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

exec bash "$REPO_ROOT/exps/commons/sbin/eval_local.sh" "$EXP_DIR"
