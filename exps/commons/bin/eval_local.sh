#!/bin/bash
# 稳定工具：从传入的实验目录读取 configs/ 下配置，启动 vLLM 并跑评估。
# 用法:
#   bash exps/commons/bin/eval_local.sh <实验目录>   # 如 exps/verl6
#   bash exps/commons/eval_local.sh <实验目录>       # 兼容包装，会转调本脚本

set -e

if [ -z "$1" ]; then
    echo "用法: $0 <实验目录>  例如: $0 exps/verl6"
    exit 1
fi
EVAL_EXP_DIR="$(cd "$1" && pwd)"
cd "$EVAL_EXP_DIR" || exit 1

if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

# 统一使用 configs 目录
mkdir -p logs
VLLM_CONFIG="configs/vllm_config4.yaml"
EVAL_CONFIG="configs/eval_config5.yaml"
VLLM_LOG="logs/vllm_server.log"
PORT=8000

echo "======================================================="
echo "🚀 Starting vLLM Server..."
echo "======================================================="

nohup uv run vllm serve --config "$VLLM_CONFIG" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "✅ vLLM Server PID: $VLLM_PID"
echo "📝 Logs are being written to: $VLLM_LOG"

cleanup() {
    echo ""
    echo "======================================================="
    echo "🧹 Cleaning up..."
    if ps -p $VLLM_PID > /dev/null 2>&1; then
        echo "🔪 Killing vLLM Server (PID: $VLLM_PID)..."
        kill $VLLM_PID
    else
        echo "⚠️ vLLM Server is not running."
    fi
    echo "👋 Done."
    echo "======================================================="
}
trap cleanup EXIT

echo "⏳ Waiting for vLLM to load model and open port $PORT..."
start_wait=$(date +%s)
timeout=600

while true; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models 2>/dev/null | grep -q "200"; then
        echo "✅ Server is up and ready!"
        break
    fi
    if ! ps -p $VLLM_PID > /dev/null 2>&1; then
        echo "❌ vLLM process died unexpectedly. Check $VLLM_LOG for details."
        exit 1
    fi
    current_time=$(date +%s)
    elapsed=$((current_time - start_wait))
    if [ $elapsed -ge $timeout ]; then
        echo "❌ Timeout waiting for server to start."
        exit 1
    fi
    sleep 5
    echo -n "."
done
echo ""

echo "======================================================="
echo "🧪 Starting Evaluation Runner..."
echo "======================================================="
EVAL_CONFIG_ABS="$EVAL_EXP_DIR/$EVAL_CONFIG"
uv run -m hardtry.utils.eval_runner "$EVAL_CONFIG_ABS"
exit 0
