#!/usr/bin/env bash

set -euo pipefail

# AIDC A 路日均数据项目级 EDA：先生成或复用日频派生数据，再独立执行 EDA。

# 共用运行体按 ROUTE 派生数据路径与结果子树；变体参数见被调脚本。
export ROUTE=A
exec "$(dirname "${BASH_SOURCE[0]}")/../_run_eda.sh" "$@"
