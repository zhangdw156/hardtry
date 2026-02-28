#!/usr/bin/env bash
# 在 CHECKPOINT_BASE 下自动找「最近一次」run 与 checkpoint，合并为完整模型并拷贝 tokenizer 等。
# 用法: merge_swift_fsdp_auto.sh <CHECKPOINT_BASE> <TARGET_DIR> [BASE_MODEL_PATH]
# 仅由实验 scripts 转调。BASE_MODEL_PATH 默认用 sft 的 base 模型（用于拷贝 tokenizer）。

set -euo pipefail

usage() {
    echo "用法: $0 <CHECKPOINT_BASE> <TARGET_DIR> [BASE_MODEL_PATH]"
    echo "示例: $0 /path/to/checkpoints/full9 /path/to/models/hardtry-4b-full9"
    exit 1
}

# 在 base 下找最近一次 run 目录：优先 v*-* 按名排序取最后，否则按 mtime 取最新
find_latest_run_dir() {
    local base="$1"
    local d
    if compgen -G "$base"/v*-* >/dev/null 2>&1; then
        ls -d "$base"/v*-* 2>/dev/null | sort -V | tail -1
        return
    fi
    ls -dt "$base"/*/ 2>/dev/null | head -1
}

# 在 run 目录下找最大的 checkpoint-<N> 的 N
find_latest_checkpoint() {
    local run_dir="$1"
    local last=""
    local d
    for d in "$run_dir"/checkpoint-*; do
        [[ -d "$d" ]] || continue
        local n="${d##*checkpoint-}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        if [[ -z "$last" || "$n" -gt "$last" ]]; then
            last="$n"
        fi
    done
    echo "$last"
}

# --- 参数 ---
CHECKPOINT_BASE="${1:-}"
TARGET_DIR="${2:-}"
BASE_MODEL_PATH="${3:-}"
[[ -n "$CHECKPOINT_BASE" && -n "$TARGET_DIR" ]] || usage
CHECKPOINT_BASE="$(cd "$CHECKPOINT_BASE" && pwd)"

RUN_DIR="$(find_latest_run_dir "$CHECKPOINT_BASE")"
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || { echo "错误: 在 $CHECKPOINT_BASE 下未找到 run 目录 (v*-* 或任意子目录)" >&2; exit 1; }
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

LAST_CKPT="$(find_latest_checkpoint "$RUN_DIR")"
[[ -n "$LAST_CKPT" ]] || { echo "错误: 在 $RUN_DIR 下未找到 checkpoint-*" >&2; exit 1; }

CKPT_PATH="$RUN_DIR/checkpoint-${LAST_CKPT}/pytorch_model_fsdp_0"
[[ -f "$CKPT_PATH" ]] || { echo "错误: 不存在 $CKPT_PATH" >&2; exit 1; }

# 若未传 BASE_MODEL_PATH，不拷贝 tokenizer（由调用方或后续步骤处理）
COPY_BASE="${BASE_MODEL_PATH:-}"

# --- 合并 ---
echo "Run:      $RUN_DIR"
echo "Checkpoint: checkpoint-${LAST_CKPT}"
echo "目标目录:   $TARGET_DIR"
accelerate merge-weights \
    "${CKPT_PATH}" \
    "${TARGET_DIR}"

# --- 拷贝 tokenizer 等（与 swift 模板 merge 脚本一致）---
if [[ -n "$COPY_BASE" && -d "$COPY_BASE" ]]; then
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt config.json configuration.json generation_config.json; do
        [[ -f "$COPY_BASE/$f" ]] && cp "$COPY_BASE/$f" "$TARGET_DIR/"
    done
    echo "已从 $COPY_BASE 拷贝 tokenizer 等至 $TARGET_DIR"
fi
