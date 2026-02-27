#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) &>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit

if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

# 配置部分
mkdir -p logs
VLLM_CONFIG="configs/vllm_config.yaml"
EVAL_CONFIG="configs/eval_config5.yaml"
VLLM_LOG="logs/vllm_server.log"
PORT=8000 # 根据你的 yaml 配置 port: 8000

echo "======================================================="
echo "🚀 Starting vLLM Server..."
echo "======================================================="

# 1. 后台启动 vLLM，并将日志重定向到文件
# nohup: 防止终端关闭导致进程退出
# &:: 在后台运行
nohup uv run vllm serve --config "$VLLM_CONFIG" > "$VLLM_LOG" 2>&1 &

# 2. 捕获 vLLM 的 PID
VLLM_PID=$!
echo "✅ vLLM Server PID: $VLLM_PID"
echo "📝 Logs are being written to: $VLLM_LOG"

# 3. 定义清理函数 (Trap)
# 无论脚本是正常结束、出错还是被 Ctrl+C 中断，都会执行这个函数
cleanup() {
    echo ""
    echo "======================================================="
    echo "🧹 Cleaning up..."
    if ps -p $VLLM_PID > /dev/null; then
        echo "🔪 Killing vLLM Server (PID: $VLLM_PID)..."
        kill $VLLM_PID
    else
        echo "⚠️ vLLM Server is not running."
    fi
    echo "👋 Done."
    echo "======================================================="
}
# 注册 trap，在 EXIT 信号（脚本退出）时触发 cleanup
trap cleanup EXIT

# 4. 健康检查：循环等待 vLLM 服务就绪
echo "⏳ Waiting for vLLM to load model and open port $PORT..."
start_wait=$(date +%s)
timeout=600 # 设置最大等待时间，例如 600秒 (10分钟)

while true; do
    # 检查端口是否通，并且返回 HTTP 200 (检查 /v1/models 接口)
    # 也可以简单用 nc -z localhost $PORT 检查端口，但 curl 更稳健（确保模型加载完）
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models | grep -q "200"; then
        echo "✅ Server is up and ready!"
        break
    fi

    # 检查进程是否意外挂掉
    if ! ps -p $VLLM_PID > /dev/null; then
        echo "❌ vLLM process died unexpectedly. Check $VLLM_LOG for details."
        exit 1
    fi

    # 超时检查
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

# 5. 运行评估脚本
echo "======================================================="
echo "🧪 Starting Evaluation Runner..."
echo "======================================================="

uv run -m hardtry.utils.eval_runner "$EVAL_CONFIG"

# 脚本运行到这里会自动触发 trap cleanup，杀死 vLLM
exit 0