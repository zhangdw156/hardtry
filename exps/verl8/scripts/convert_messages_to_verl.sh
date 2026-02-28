#!/usr/bin/env bash
# 步骤：调用 hardtry.utils.convert_messages_to_verl，使用本实验 configs 下配置。可从任意目录执行。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
CONFIG="$EXP_DIR/configs/convert_messages_to_verl_config.yaml"

cd "$REPO_ROOT" || exit 1
uv run python -m hardtry.utils.convert_messages_to_verl "$CONFIG"
