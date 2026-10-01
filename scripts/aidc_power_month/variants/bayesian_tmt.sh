#!/usr/bin/env bash
# bayesian_tmt 是实验性单序列贝叶斯滞后回归近似，保留历史 registry 名称。
# 变体定义：bayesian_tmt（模型 bayesian_tmt）；公共运行体在 ../_common.sh。
MODEL_NAME=bayesian_tmt
MODEL_PARAMS='{"lags":[1,2,7]}'

_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_source_dir}/../_common.sh"
run_single_model
