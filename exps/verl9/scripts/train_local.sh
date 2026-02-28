#!/bin/bash
# 步骤：GRPO 训练。可从任意目录执行（仅依赖脚本自身路径，执行前会 cd 到实验目录）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
cd "$EXP_DIR" || exit 1

VERL_VENV="/dfs/data/uv-venv/verl"
PYTHON_CMD="$VERL_VENV/bin/python3"
CONFIG_PATH="/dfs/data/work/hardtry/exps/${EXP_NAME}/configs"
CONFIG_NAME="grpo_config"
mkdir -p "$EXP_DIR/logs"
LOG_PATH="$EXP_DIR/logs/grpo.log"

# 确保本脚本及其子进程（含 Ray worker）都使用 verl 环境，避免 Python 版本不一致
export PATH="$VERL_VENV/bin:$PATH"
export VIRTUAL_ENV="$VERL_VENV"

echo "EXP_DIR: $EXP_DIR"
${PYTHON_CMD} -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" | tee "$LOG_PATH"
