#!/bin/bash
# 唯一入口：从仓库根目录执行一条龙。配置在 configs/，步骤脚本在 scripts/。
# 模板占位符 __EXP_NAME__ 会在 new_exp.sh 复制时被替换。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

uv run python -m hardtry.run config_file=exps/__EXP_NAME__/configs/run___EXP_NAME__.yaml
