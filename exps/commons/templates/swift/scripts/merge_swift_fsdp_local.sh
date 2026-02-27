#!/bin/bash
# 步骤：Swift/FSDP checkpoint 合并为完整模型。可从任意目录执行（使用绝对路径）。复制后请将 CKPT_PATH 改为本次训练的 checkpoint 路径。

source /dfs/data/uv-venv/huggingface/bin/activate

BASE_MODEL_PATH="/dfs/data/models/Qwen3-4B-Instruct-2507"
# 复制后修改：替换为 exps/<实验名> 下本次训练产生的路径，例如
# .../checkpoints/__EXP_NAME__/v0-20260227-133416/checkpoint-11/pytorch_model_fsdp_0
CKPT_PATH="/dfs/data/work/hardtry/checkpoints/__EXP_NAME__/v0-YYYYMMDD-HHMMSS/checkpoint-N/pytorch_model_fsdp_0"
OUTPUT_PATH="/dfs/data/models/hardtry-4b-__EXP_NAME__"

accelerate merge-weights \
    "${CKPT_PATH}" \
    "${OUTPUT_PATH}"

cp "${BASE_MODEL_PATH}/tokenizer.json" "${OUTPUT_PATH}"
cp "${BASE_MODEL_PATH}/tokenizer_config.json" "${OUTPUT_PATH}"
cp "${BASE_MODEL_PATH}/vocab.json" "${OUTPUT_PATH}"
cp "${BASE_MODEL_PATH}/merges.txt" "${OUTPUT_PATH}"
cp "${BASE_MODEL_PATH}/config.json" "${OUTPUT_PATH}"
cp "${BASE_MODEL_PATH}/configuration.json" "${OUTPUT_PATH}"
cp "${BASE_MODEL_PATH}/generation_config.json" "${OUTPUT_PATH}"
