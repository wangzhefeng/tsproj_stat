#!/usr/bin/env bash

set -euo pipefail

# AIDC 负荷（B路，日均）单变量脚本：从 dataset/aidc_power_month/B_Loads_5min_20251001_20260728.csv 聚合生成 derived/B_Loads_1day_mean_20251001_20260728.csv，time 为时间列，value 为目标列。
# 分支B 对照实验：线性去趋势(detrend_method=linear) + 平稳残差 AR(1)。与"差分 ARIMA"分支A对照，回答 EDA 第8节的核心问题（差分 vs 线性去趋势哪个更稳），不同时叠加两种趋势处理。

# 共用运行体按 ROUTE 派生数据路径与结果子树；变体参数见被调脚本。
export ROUTE=B
exec "$(dirname "${BASH_SOURCE[0]}")/../../variants/ar_detrend_p1.sh" "$@"
