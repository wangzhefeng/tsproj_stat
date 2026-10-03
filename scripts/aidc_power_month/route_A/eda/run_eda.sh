#!/usr/bin/env bash

set -euo pipefail

# AIDC A 路只读 EDA，默认日频；--config 可选择15分钟或小时场景。

# 场景入口只选择配置，通用运行逻辑归 eda/runner.py。
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
export LOG_NAME=eda_A_Loads
exec .venv/bin/python -u run_eda.py --config scripts/aidc_power_month/route_A/eda/D.yaml "$@"
