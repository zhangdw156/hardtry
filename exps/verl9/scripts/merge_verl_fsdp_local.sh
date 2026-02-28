#!/bin/bash
# 步骤：使用「最后一次保存」的 checkpoint 自动 merge。可从任意目录执行（仅依赖脚本自身路径）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts 在 exps/<实验名>/scripts/，上三级才是仓库根
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 与 verl_config.checkpoint_dir / merged_model_path 一致，使用相同占位符
CHECKPOINT_BASE="/dfs/data/work/hardtry/checkpoints/verl9"
TARGET_DIR="/dfs/data/models/hardtry-4b-verl9"

exec bash "$REPO_ROOT/exps/commons/sbin/merge_verl_fsdp_auto.sh" "$CHECKPOINT_BASE" "$TARGET_DIR"
