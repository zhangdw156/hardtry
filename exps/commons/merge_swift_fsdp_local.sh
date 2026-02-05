#!/bin/bash

source /dfs/data/uv-venv/huggingface/bin/activate

BASE_MODEL_PATH="/dfs/data/models/Qwen3-4B-Instruct-2507"
##############################
# TODO: To be modified
##############################
CKPT_PATH="/dfs/data/work/hardtry/checkpoints/full1/v6-20260204-133724/checkpoint-162/pytorch_model_fsdp_0"
##############################
# TODO: To be modified
##############################
OUTPUT_PATH="/dfs/data/models/hardtry-4b-full1"

accelerate merge-weights \
    ${CKPT_PATH} \
    ${OUTPUT_PATH}

cp ${BASE_MODEL_PATH}/tokenizer.json ${OUTPUT_PATH}
cp ${BASE_MODEL_PATH}/tokenizer_config.json ${OUTPUT_PATH}
cp ${BASE_MODEL_PATH}/vocab.json ${OUTPUT_PATH}
cp ${BASE_MODEL_PATH}/merges.txt ${OUTPUT_PATH}
cp ${BASE_MODEL_PATH}/config.json ${OUTPUT_PATH}
cp ${BASE_MODEL_PATH}/configuration.json ${OUTPUT_PATH}
cp ${BASE_MODEL_PATH}/generation_config.json ${OUTPUT_PATH}
