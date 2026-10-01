#!/usr/bin/env bash
# RAR 是实验性残差自回归模型。
# 变体定义：rar（模型 rar）；公共运行体在 ../_common.sh。
MODEL_NAME=rar
MODEL_PARAMS='{"alpha":0.2}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
