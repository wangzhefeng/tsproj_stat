#!/usr/bin/env bash
# seasonal_naive 使用日频周周期，作为低成本季节基线。
# 变体定义：seasonal_naive（模型 seasonal_naive）；公共运行体在 ../_common.sh。
MODEL_NAME=seasonal_naive
MODEL_PARAMS='{"season_length":7}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
