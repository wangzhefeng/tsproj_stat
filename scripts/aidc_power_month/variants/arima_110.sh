#!/usr/bin/env bash
# ARIMA [1,1,0]：一阶差分(d=1)后纯 AR(1)，最简约的 AR 主导结构，契合 PACF 滞后1后截尾的诊断。
# 变体定义：arima_110（模型 arima）；公共运行体在 ../_common.sh。
MODEL_NAME=arima
MODEL_PARAMS='{"order":[1,1,0]}'
LOG_NAME_OPT="arima_110"

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
