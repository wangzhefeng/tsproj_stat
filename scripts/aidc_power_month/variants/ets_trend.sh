#!/usr/bin/env bash
# ETS 仅趋势项（Holt 线性）：去掉弱季节项，专注趋势；与 trend+seasonal(7) 的默认配置对照季节项是否有收益。
# 变体定义：ets_trend（模型 ets）；公共运行体在 ../_common.sh。
MODEL_NAME=ets
MODEL_PARAMS='{"trend":"add"}'
LOG_NAME_OPT="ets_trend"

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
