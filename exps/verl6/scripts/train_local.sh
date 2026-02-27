#!/bin/bash
# 从 scripts/ 运行时，实验目录为脚本所在目录的上级。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"

PYTHON_CMD="/dfs/data/uv-venv/verl/bin/python3"
CONFIG_PATH="/dfs/data/work/hardtry/exps/${EXP_NAME}/configs"
CONFIG_NAME="grpo_config"
mkdir -p "$EXP_DIR/logs"
LOG_PATH="$EXP_DIR/logs/grpo.log"

echo "EXP_DIR: $EXP_DIR"
${PYTHON_CMD} -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" | tee "$LOG_PATH"
