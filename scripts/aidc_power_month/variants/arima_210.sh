#!/usr/bin/env bash
# ARIMA [2,1,0]：一阶差分(d=1)后 AR(2)，捕捉差分序列中略长的自相关结构，与 [1,1,0] 对照阶数敏感性。
# 变体定义：arima_210（模型 arima）；公共运行体在 ../_common.sh。
MODEL_NAME=arima
MODEL_PARAMS='{"order":[2,1,0]}'
LOG_NAME_OPT="arima_210"

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
