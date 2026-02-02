#!/bin/bash

SCRIPT_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd)"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

PYTHON_CMD="/dfs/data/uv-venv/verl/bin/python3"

${PYTHON_CMD} -m verl.trainer.main_ppo \
    --config-path="/dfs/data/work/hardtry/exps/verl1/configs" \
    --config-name="grpo_lora_config" | tee ${SCRIPT_DIR}/logs/grpo_lora.log