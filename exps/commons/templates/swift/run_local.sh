#!/bin/bash
# 唯一入口：在实验目录下按顺序执行各步骤脚本，不再使用 run.py。
# 模板占位符 __EXP_NAME__ 由 new_exp.sh 替换。

set -e
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP_DIR" || exit 1

if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

echo "=========================================="
echo "步骤 1/3: train (SFT)"
echo "=========================================="
bash scripts/train_local.sh

echo "=========================================="
echo "步骤 2/3: merge"
echo "=========================================="
bash scripts/merge_swift_fsdp_local.sh

echo "=========================================="
echo "步骤 3/3: eval"
echo "=========================================="
bash scripts/eval_local.sh

echo "=========================================="
echo "全部完成"
echo "=========================================="
