#!/usr/bin/env bash
# SARIMA 季节差分关闭(D=0)：seasonal_order=[1,0,1,7]，对应 EDA 的 D=0 建议；保留周季节 AR/MA 项但不做季节差分，避免弱季节性下过度差分。
# 变体定义：sarima_D0（模型 sarima）；公共运行体在 ../_common.sh。
MODEL_NAME=sarima
MODEL_PARAMS='{"order":[1,1,1],"seasonal_order":[1,0,1,7],"enforce_stationarity":false,"enforce_invertibility":false,"fit_kwargs":{"disp":false,"maxiter":20}}'
BACKTEST_VERBOSE=true
LOG_NAME_OPT="sarima_D0"

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
