#!/usr/bin/env bash
# naive 是最后值延续基线，用于快速检查数据、CLI 和结果落盘链路。
# 变体定义：naive（模型 naive）；公共运行体在 ../_common.sh。
MODEL_NAME=naive
MODEL_PARAMS='{}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
