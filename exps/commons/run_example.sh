#!/bin/bash
# commons 示例：依次执行两段数据转换（hardgen → messages → verl parquet）。
# 可从任意目录执行；会 cd 到仓库根再运行。
# 用法: bash exps/commons/run_example.sh

set -e
COMMONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMONS_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

echo "步骤 1/2: convert_hardgen_to_messages"
uv run python -m hardtry.utils.convert_hardgen_to_messages \
    exps/commons/configs/convert_hardgen_to_messages_config.yaml

echo "步骤 2/2: convert_messages_to_verl"
uv run python -m hardtry.utils.convert_messages_to_verl \
    exps/commons/configs/convert_messages_to_verl_config.yaml

echo "完成"
