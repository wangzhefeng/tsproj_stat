#!/usr/bin/env bash

set -euo pipefail

# AIDC 负荷（A路，日均）单变量脚本：从 dataset/aidc_power_month/A_Loads_5min_20251001_20260728.csv 聚合生成 derived/A_Loads_1day_mean_20251001_20260728.csv，time 为时间列，value 为目标列。
# 分支B 对照实验：线性去趋势(detrend_method=linear) + 平稳残差 AR(2)。提高 AR 阶数，检验去趋势后残差是否需要更长记忆。

# 共用运行体按 ROUTE 派生数据路径与结果子树；变体参数见被调脚本。
export ROUTE=A
exec "$(dirname "${BASH_SOURCE[0]}")/../../variants/ar_detrend_p2.sh" "$@"
