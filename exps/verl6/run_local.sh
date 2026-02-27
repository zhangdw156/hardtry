#!/bin/bash

SCRIPT_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd)"

cd "${SCRIPT_DIR}/../../"

if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

uv run python -m hardtry.run config_file=exps/verl6/run_verl6.yaml