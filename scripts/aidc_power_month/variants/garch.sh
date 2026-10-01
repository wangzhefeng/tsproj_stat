#!/usr/bin/env bash
# GARCH 属于 optional 波动率模型，依赖 arch 包。
# 变体定义：garch（模型 garch）；公共运行体在 ../_common.sh。
MODEL_NAME=garch
MODEL_PARAMS='{}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
