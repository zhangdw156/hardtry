#!/bin/bash

SCRIPT_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd)"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

if [ -n "$USE_LOCAL_VERL" ]; then
    PYTHON_CMD="/dfs/data/uv-venv/verl/bin/python3"
else
    PYTHON_CMD="python3"
fi

${PYTHON_CMD} -m verl.trainer.main_ppo \
    --config-path="/dfs/data/work/hardtry/exps/verl1/configs" \
    --config-name="grpo_config" | tee logs/verl.log