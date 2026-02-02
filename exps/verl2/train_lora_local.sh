#!/bin/bash

SCRIPT_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd)"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

PYTHON_CMD="/dfs/data/uv-venv/verl/bin/python3"
CONFIG_PATH="/dfs/data/work/hardtry/exps/verl2/configs"
CONFIG_NAME="grpo_lora_config"
LOG_PATH="${SCRIPT_DIR}/logs/grpo_lora.log"

${PYTHON_CMD} -m verl.trainer.main_ppo \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} | tee ${LOG_PATH}