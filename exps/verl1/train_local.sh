#!/bin/bash

SCRIPT_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd)"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

EXP_NAME="$(basename "${SCRIPT_DIR}")"
PYTHON_CMD="/dfs/data/uv-venv/verl/bin/python3"
CONFIG_PATH="/dfs/data/work/hardtry/exps/${EXP_NAME}/configs"
CONFIG_NAME="grpo_config"
LOG_PATH="${SCRIPT_DIR}/logs/grpo.log"

export RAY_DISABLE_DASHBOARD=1
export WORLD_SIZE=4
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VE_OMP_NUM_THREADS=8

${PYTHON_CMD} -m verl.trainer.main_ppo \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} | tee ${LOG_PATH}