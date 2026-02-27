#!/bin/bash
# 唯一入口：从仓库根目录执行一条龙，配置在 configs/，步骤脚本在 scripts/。

SCRIPT_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd)"

cd "${SCRIPT_DIR}/../../"

if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

uv run python -m hardtry.run config_file=exps/verl6/configs/run_verl6.yaml