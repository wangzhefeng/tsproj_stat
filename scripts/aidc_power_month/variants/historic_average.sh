#!/usr/bin/env bash
# historic_average 使用最近约半年均值，作为低成本均值基线。
# 变体定义：historic_average（模型 historic_average）；公共运行体在 ../_common.sh。
MODEL_NAME=historic_average
MODEL_PARAMS='{"window":180}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
