#!/usr/bin/env bash
# dynamic_theta 属于 optional StatsForecast 模型，显式传入日频和周季节长度。
# 变体定义：dynamic_theta（模型 dynamic_theta）；公共运行体在 ../_common.sh。
MODEL_NAME=dynamic_theta
MODEL_PARAMS='{"season_length":7,"freq":"D"}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
