#!/usr/bin/env bash

set -euo pipefail

# 风电数据项目级 EDA：每份数据单独运行一次，不执行模型训练、回测或预测。
cd "$(dirname "$0")/../../.."

export LOG_NAME=eda_wind_dataset
exec .venv/bin/python -u run_eda.py --config scripts/wind/eda/D.yaml "$@"
