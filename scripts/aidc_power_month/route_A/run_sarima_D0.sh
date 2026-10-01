#!/usr/bin/env bash

set -euo pipefail

# AIDC 负荷（A路，日均）单变量脚本：从 dataset/aidc_power_month/A_Loads_5min_20251001_20260728.csv 聚合生成 derived/A_Loads_1day_mean_20251001_20260728.csv，time 为时间列，value 为目标列。
# SARIMA 季节差分关闭(D=0)：seasonal_order=[1,0,1,7]，对应 EDA 的 D=0 建议；保留周季节 AR/MA 项但不做季节差分，避免弱季节性下过度差分。

# 共用运行体按 ROUTE 派生数据路径与结果子树；变体参数见被调脚本。
export ROUTE=A
exec "$(dirname "${BASH_SOURCE[0]}")/../variants/sarima_D0.sh" "$@"
