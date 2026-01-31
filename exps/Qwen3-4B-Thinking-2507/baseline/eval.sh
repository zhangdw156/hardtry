#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取当前目录的名称 (不包含上级路径)
CURRENT_DIR_NAME=$(basename "$SCRIPT_DIR")

source /dfs/data/uv-venv/gorilla/bin/activate

MODEL_NAME="Qwen/Qwen3-4B-FC"
TEST_CATEGORY="multi_turn_base"
THREADS=16
ARTIFACT_DIR="${SCRIPT_DIR}/../../eval_results/${CURRENT_DIR_NAME}"

# =========================================================
# 手动导出环境变量，强制让 BFCL 识别本地 vLLM
# =========================================================
export REMOTE_OPENAI_BASE_URL="http://localhost:8000/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
export REMOTE_OPENAI_TOKENIZER_PATH="/dfs/data/models/Qwen3-4B-Thinking-2507"

echo "======================================================="
echo "🚀 开始任务"
echo "📂 脚本位置: $SCRIPT_DIR"
echo "📂 结果输出: $ARTIFACT_DIR"
echo "🤖 模型名称: $MODEL_NAME"
echo "📋 测试类别: $TEST_CATEGORY"
echo "======================================================="

# =========================================================
# 执行生成 (Generate)
# =========================================================
echo "▶️ [1/2] Running Generation..."
bfcl generate \
    --model "$MODEL_NAME" \
    --test-category "$TEST_CATEGORY" \
    --backend vllm \
    --skip-server-setup \
    --num-threads "$THREADS" \
    --result-dir "$ARTIFACT_DIR/result"

# =========================================================
# 执行评测 (Evaluate)
# =========================================================
echo "▶️ [2/2] Running Evaluation..."
bfcl evaluate \
  --model "$MODEL_NAME" \
  --test-category "$TEST_CATEGORY" \
  --result-dir "$ARTIFACT_DIR/result" \
  --score-dir "$ARTIFACT_DIR/score"

echo "✅ 所有任务完成！"