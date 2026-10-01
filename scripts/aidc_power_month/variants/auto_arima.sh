#!/usr/bin/env bash
# auto_arima 搜索成本较高，本脚本默认采用偏快的日常参数集。
# 变体定义：auto_arima（模型 auto_arima）；公共运行体在 ../_common.sh。
MODEL_NAME=auto_arima
MODEL_PARAMS='{"seasonal": false, "m": 1, "stepwise": true, "start_p": 0, "start_q": 0, "max_p": 2, "max_q": 2, "max_order": 4, "maxiter": 20, "information_criterion": "aic", "trace": true, "error_action": "ignore", "suppress_warnings": true}'
BACKTEST_VERBOSE=true

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
