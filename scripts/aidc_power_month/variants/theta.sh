#!/usr/bin/env bash
# theta 是轻量统计预测模型，这里显式设置日频周周期。
# 变体定义：theta（模型 theta）；公共运行体在 ../_common.sh。
MODEL_NAME=theta
MODEL_PARAMS='{"period":7}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
