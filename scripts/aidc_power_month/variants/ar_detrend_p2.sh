#!/usr/bin/env bash
# 分支B 对照实验：线性去趋势(detrend_method=linear) + 平稳残差 AR(2)。提高 AR 阶数，检验去趋势后残差是否需要更长记忆。
# 变体定义：ar_detrend_p2（模型 ar）；公共运行体在 ../_common.sh。
MODEL_NAME=ar
MODEL_PARAMS='{"p":2}'
DETREND_METHOD=linear
LOG_NAME_OPT="ar_detrend_p2"

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
