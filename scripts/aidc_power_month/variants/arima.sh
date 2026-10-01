#!/usr/bin/env bash
# ARIMA 使用固定低阶 (1,1,1)，适合作为日常统计模型基线。
# 变体定义：arima（模型 arima）；公共运行体在 ../_common.sh。
MODEL_NAME=arima
MODEL_PARAMS='{"order":[1,1,1]}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
