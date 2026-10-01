#!/usr/bin/env bash
# Theta period=1：纯趋势无季节，契合周季节性弱（强度0.096）的结构；Theta 方法的甜区即近线性趋势。
# 变体定义：theta_p1（模型 theta）；公共运行体在 ../_common.sh。
MODEL_NAME=theta
MODEL_PARAMS='{"period":1}'
LOG_NAME_OPT="theta_p1"

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
