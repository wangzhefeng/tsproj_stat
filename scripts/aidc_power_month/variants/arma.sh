#!/usr/bin/env bash
# ARMA 是 ARIMA 家族的无差分 ARMA(1,1) 入口。
# 变体定义：arma（模型 arma）；公共运行体在 ../_common.sh。
MODEL_NAME=arma
MODEL_PARAMS='{"p":1,"q":1}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
