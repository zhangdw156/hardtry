#!/bin/bash
# 步骤：将 messages 转为 verl 所需 parquet。可从任意目录执行（仅依赖脚本自身路径）。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
CONFIG="$EXP_DIR/configs/convert_messages_to_verl_config.yaml"

cd "$REPO_ROOT" || exit 1
uv run python -m hardtry.utils.convert_messages_to_verl "$CONFIG"
