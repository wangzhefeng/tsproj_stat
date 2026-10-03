#!/usr/bin/env bash

set -euo pipefail

# ETTm1 数据集独立 EDA；仅读取现有 CSV，不执行训练/回测/预测。
cd "$(dirname "$0")/../../../.."
export LOG_NAME=eda_ETTm1
exec .venv/bin/python -u run_eda.py --config scripts/ett_small/ETTm1/eda/15min.yaml "$@"
