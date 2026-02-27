#!/bin/bash
# 兼容包装：转调 bin/eval_local.sh。无参时使用 commons 的 configs，有参时使用指定实验目录。
COMMONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMONS_DIR/../.." && pwd)"
if [ -n "$1" ]; then
    exec bash "$REPO_ROOT/exps/commons/bin/eval_local.sh" "$1"
else
    exec bash "$REPO_ROOT/exps/commons/bin/eval_local.sh" "$COMMONS_DIR"
fi
