#!/usr/bin/env bash
# 步骤：转调 exps/commons/bin/merge_verl_fsdp_auto.sh 做 merge。可从任意目录执行。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts 在 exps/<实验名>/scripts/，上三级才是仓库根
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CHECKPOINT_BASE="/dfs/data/work/hardtry/checkpoints/verl7"
TARGET_DIR="/dfs/data/models/hardtry-4b-verl7"

exec bash "$REPO_ROOT/exps/commons/bin/merge_verl_fsdp_auto.sh" "$CHECKPOINT_BASE" "$TARGET_DIR"
