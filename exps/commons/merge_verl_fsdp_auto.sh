#!/bin/bash
# 兼容包装：转调 bin/merge_verl_fsdp_auto.sh，参数原样传递。
COMMONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMONS_DIR/../.." && pwd)"
exec bash "$REPO_ROOT/exps/commons/bin/merge_verl_fsdp_auto.sh" "$@"
