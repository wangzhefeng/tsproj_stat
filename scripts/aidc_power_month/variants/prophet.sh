#!/usr/bin/env bash
# prophet 作为 optional 扩展模型接入统一 fit/predict 契约。
# 变体定义：prophet（模型 prophet）；公共运行体在 ../_common.sh。
MODEL_NAME=prophet
MODEL_PARAMS='{"freq":"D"}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
