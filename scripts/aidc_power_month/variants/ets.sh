#!/usr/bin/env bash
# ETS 是指数平滑统一入口，这里启用日频周季节项。
# 变体定义：ets（模型 ets）；公共运行体在 ../_common.sh。
MODEL_NAME=ets
MODEL_PARAMS='{"trend":"add","seasonal":"add","seasonal_periods":7}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
