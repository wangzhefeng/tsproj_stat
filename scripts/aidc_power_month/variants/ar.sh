#!/usr/bin/env bash
# AR 是 ARIMA 家族的自回归特例。
# 变体定义：ar（模型 ar）；公共运行体在 ../_common.sh。
MODEL_NAME=ar
MODEL_PARAMS='{"p":2}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
