#!/usr/bin/env bash
# 步骤：转调 exps/commons/bin/eval_local.sh 启动 vLLM 并跑评估。可从任意目录执行。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# scripts 在 exps/<实验名>/scripts/，上三级才是仓库根
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

exec bash "$REPO_ROOT/exps/commons/bin/eval_local.sh" "$EXP_DIR"
