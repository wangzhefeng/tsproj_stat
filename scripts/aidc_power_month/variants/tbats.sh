#!/usr/bin/env bash
# TBATS 属于 optional 扩展模型；当前 smoke 可运行，但可能因依赖/runtime 问题走 ETS fallback。
# 变体定义：tbats（模型 tbats）；公共运行体在 ../_common.sh。
MODEL_NAME=tbats
MODEL_PARAMS='{"seasonal_periods":[7],"show_warnings":false,"n_jobs":1}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
