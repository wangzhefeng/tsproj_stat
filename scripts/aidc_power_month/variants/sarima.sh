#!/usr/bin/env bash
# SARIMA 拟合成本较高，本脚本默认采用偏快的日常参数集。
# 变体定义：sarima（模型 sarima）；公共运行体在 ../_common.sh。
MODEL_NAME=sarima
MODEL_PARAMS='{"order":[1,1,1],"seasonal_order":[1,1,1,7],"enforce_stationarity":false,"enforce_invertibility":false,"fit_kwargs":{"disp":false,"maxiter":20}}'
BACKTEST_VERBOSE=true

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
