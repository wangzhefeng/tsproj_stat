#!/usr/bin/env bash
# MA 是 ARIMA 家族的移动平均特例。
# 变体定义：ma（模型 ma）；公共运行体在 ../_common.sh。
MODEL_NAME=ma
MODEL_PARAMS='{"q":1}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
