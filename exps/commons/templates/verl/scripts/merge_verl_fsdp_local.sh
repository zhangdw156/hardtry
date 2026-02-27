#!/bin/bash
# 使用「最后一次保存」的 checkpoint 自动 merge。路径中的 __EXP_NAME__ 由 new_exp.sh 替换。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts 在 exps/<实验名>/scripts/，上三级才是仓库根
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CHECKPOINT_BASE="/dfs/data/work/hardtry/checkpoints/__EXP_NAME__"
TARGET_DIR="/dfs/data/models/hardtry-4b-__EXP_NAME__"

exec bash "$REPO_ROOT/exps/commons/bin/merge_verl_fsdp_auto.sh" "$CHECKPOINT_BASE" "$TARGET_DIR"
