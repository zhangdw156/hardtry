#!/usr/bin/env bash
# 在指定实验目录下启动 vLLM，再执行 BFCL 评估。实验目录需含 configs/vllm_config.yaml 与 configs/eval_config5.yaml。
# 用法: eval_local.sh <实验目录>

set -euo pipefail

readonly VLLM_PORT=8000
readonly VLLM_TIMEOUT=600
readonly EVAL_CONFIG_REL="configs/eval_config5.yaml"
readonly VLLM_CONFIG_REL="configs/vllm_config.yaml"
readonly VLLM_LOG_REL="logs/vllm_server.log"

usage() {
    echo "用法: $0 <实验目录>"
    echo "示例: $0 exps/verl7"
    exit 1
}

# 等待 vLLM 在 PORT 上就绪
wait_for_vllm() {
    local pid=$1
    local port=$2
    local timeout=$3
    local start now elapsed
    start=$(date +%s)
    while true; do
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q "200"; then
            echo "✅ vLLM 已就绪"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "❌ vLLM 进程已退出，请查看日志" >&2
            return 1
        fi
        now=$(date +%s)
        elapsed=$((now - start))
        if (( elapsed >= timeout )); then
            echo "❌ 等待 vLLM 超时 (${timeout}s)" >&2
            return 1
        fi
        sleep 5
        echo -n "."
    done
}

# --- 参数 ---
[[ -n "${1:-}" ]] || usage
EXP_DIR="$(cd "$1" && pwd)"
cd "$EXP_DIR"

VLLM_CONFIG="$EXP_DIR/$VLLM_CONFIG_REL"
EVAL_CONFIG_ABS="$EXP_DIR/$EVAL_CONFIG_REL"
VLLM_LOG="$EXP_DIR/$VLLM_LOG_REL"

[[ -f "$VLLM_CONFIG" ]] || { echo "错误: 未找到 $VLLM_CONFIG_REL" >&2; exit 1; }
[[ -f "$EVAL_CONFIG_ABS" ]] || { echo "错误: 未找到 $EVAL_CONFIG_REL" >&2; exit 1; }

mkdir -p logs
[[ -f /dfs/data/sbin/setup.sh ]] && source /dfs/data/sbin/setup.sh

# --- 启动 vLLM ---
echo "======================================================="
echo "🚀 启动 vLLM..."
echo "======================================================="
nohup uv run vllm serve --config "$VLLM_CONFIG" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "PID: $VLLM_PID  日志: $VLLM_LOG"

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "🛑 停止 vLLM (PID $VLLM_PID)"
        kill "$VLLM_PID"
    fi
}
trap cleanup EXIT

echo "⏳ 等待 vLLM 就绪 (port $VLLM_PORT)..."
wait_for_vllm "$VLLM_PID" "$VLLM_PORT" "$VLLM_TIMEOUT"
echo ""

# --- 评估 ---
echo "======================================================="
echo "🧪 运行评估..."
echo "======================================================="
uv run -m hardtry.utils.eval_runner "$EVAL_CONFIG_ABS"
