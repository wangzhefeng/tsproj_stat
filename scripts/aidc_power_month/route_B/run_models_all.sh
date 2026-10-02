#!/usr/bin/env bash

set -euo pipefail

# 模型清单、超参和阶段开关见 ../_run_models_all.sh；本入口不执行 EDA。

# 共用运行体按 ROUTE 派生数据路径与结果子树；变体参数见被调脚本。
export ROUTE=B
exec "$(dirname "${BASH_SOURCE[0]}")/../_run_models_all.sh" "$@"
