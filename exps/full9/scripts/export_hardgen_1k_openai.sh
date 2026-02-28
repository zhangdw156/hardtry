#!/bin/bash
# train.parquet → train.jsonl，test.parquet → test.jsonl（与 RL 同划分，SFT 中 dataset/val_dataset 分开指定）。
# full8/full9 运行前必须先执行本脚本，否则 sft_config 中的 train.jsonl / test.jsonl 不存在。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/hardgen_1k"

if [[ ! -f "${DATA_DIR}/train.parquet" ]] || [[ ! -f "${DATA_DIR}/test.parquet" ]]; then
    echo "错误: 未找到 ${DATA_DIR}/train.parquet 或 test.parquet，请先运行 verl7/verl8 的 convert 生成 hardgen_1k。" >&2
    exit 1
fi

cd "$REPO_ROOT"
uv run python -m hardtry.utils.parquet_to_openai_messages \
    --input_dir "$DATA_DIR" \
    --output_dir "$DATA_DIR"

echo "完成。已生成 train.jsonl（对应 train.parquet）、test.jsonl（对应 test.parquet）"
