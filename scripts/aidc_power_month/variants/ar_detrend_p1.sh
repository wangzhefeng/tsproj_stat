#!/usr/bin/env bash
# 分支B 对照实验：线性去趋势(detrend_method=linear) + 平稳残差 AR(1)。与"差分 ARIMA"分支A对照，回答 EDA 第8节的核心问题（差分 vs 线性去趋势哪个更稳），不同时叠加两种趋势处理。
# 变体定义：ar_detrend_p1（模型 ar）；公共运行体在 ../_common.sh。
MODEL_NAME=ar
MODEL_PARAMS='{"p":1}'
DETREND_METHOD=linear
LOG_NAME_OPT="ar_detrend_p1"

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
