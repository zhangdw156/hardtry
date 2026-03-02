#!/bin/bash
# 仅执行 train → merge → eval，不执行 convert。
# 用于与 verl11/verl12 共数据：先执行一次 convert（verl10/verl11/verl12 任选），再对本实验执行 run_train_only.sh。
# 使用前请确保已生成数据：bash exps/verl10/scripts/convert_messages_to_verl.sh（输出 data/hardgen_1k_shuffle）。

set -e
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP_DIR" || exit 1

if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

echo "=========================================="
echo "步骤 1/3: train (GRPO)"
echo "=========================================="
bash scripts/train_local.sh

echo "=========================================="
echo "步骤 2/3: merge"
echo "=========================================="
bash scripts/merge_verl_fsdp_local.sh

echo "=========================================="
echo "步骤 3/3: eval"
echo "=========================================="
bash scripts/eval_local.sh

echo "=========================================="
echo "全部完成（未执行 convert）"
echo "=========================================="
