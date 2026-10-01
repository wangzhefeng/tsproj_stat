#!/usr/bin/env bash
# croston 是 experimental 间歇需求基线；负荷非负时可用于对照。
# 变体定义：croston（模型 croston）；公共运行体在 ../_common.sh。
MODEL_NAME=croston
MODEL_PARAMS='{"alpha":0.2}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
