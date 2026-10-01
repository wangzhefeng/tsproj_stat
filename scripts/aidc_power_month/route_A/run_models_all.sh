#!/usr/bin/env bash

set -euo pipefail

# 每模型超参见 --batch_models；neuralprophet 因环境损坏未纳入（原脚本保留）；
# 参数变体对照（arima_110/210、ets_trend、sarima_D0、theta_p1、ar_detrend_*）参数轴不同，保留独立脚本。
# 数据加载/聚合/预处理/EDA 只做一次；各模型独立 experiment_path；对比表 results_{data_name}/results_test/comparison/model_comparison.csv。

# 共用运行体按 ROUTE 派生数据路径与结果子树；变体参数见被调脚本。
export ROUTE=A
exec "$(dirname "${BASH_SOURCE[0]}")/../_run_models_all.sh" "$@"
