#!/bin/bash
# 使用「最后一次保存」的 checkpoint 自动 merge，无需手改步数；可被 run 一条龙调用。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT_BASE="/dfs/data/work/hardtry/checkpoints/verl6"
TARGET_DIR="/dfs/data/models/hardtry-4b-verl6"

exec bash "$REPO_ROOT/exps/commons/merge_verl_fsdp_auto.sh" "$CHECKPOINT_BASE" "$TARGET_DIR"
