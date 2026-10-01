#!/usr/bin/env bash
# NeuralProphet 属于 experimental 扩展模型；当前 smoke 可运行，但可能因 holidays 兼容问题走趋势 fallback。
# 变体定义：neuralprophet（模型 neuralprophet）；公共运行体在 ../_common.sh。
MODEL_NAME=neuralprophet
MODEL_PARAMS='{"freq":"D"}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
