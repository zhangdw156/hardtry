#!/bin/bash
# 使用「最后一次保存」的 checkpoint 自动 merge，可被 run 一条龙调用。
# 实验目录为脚本所在目录的上级。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts 在 exps/verl6/scripts/，上三级才是仓库根
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CHECKPOINT_BASE="/dfs/data/work/hardtry/checkpoints/verl6"
TARGET_DIR="/dfs/data/models/hardtry-4b-verl6"

exec bash "$REPO_ROOT/exps/commons/bin/merge_verl_fsdp_auto.sh" "$CHECKPOINT_BASE" "$TARGET_DIR"
