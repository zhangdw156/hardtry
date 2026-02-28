#!/bin/bash
# 仅做训练一步，用于验证 VeRL 可跑通且奖励函数被正确加载并应用。观察日志中的 [verl7_demo reward_fn] 输出。

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

export PATH="$VERL_VENV/bin:$PATH"
export VIRTUAL_ENV="$VERL_VENV"

echo "EXP_DIR: $EXP_DIR"
echo "CONFIG: $CONFIG_PATH / $CONFIG_NAME"
${PYTHON_CMD} -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" | tee "$LOG_PATH"
