#!/usr/bin/env bash
# AIDC 路日均数据项目级 EDA 共用体：先生成或复用日频派生数据，再独立执行 EDA。
# 由 route_A/route_B/run_eda.sh 设置 ROUTE 后 exec。
set -euo pipefail
: "${ROUTE:?ROUTE must be A or B}"

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${_script_dir}/../.."

model_name=naive
export LOG_NAME=eda_${ROUTE}_Loads_1day

.venv/bin/python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path "dataset/aidc_power_month/${ROUTE}_Loads_5min_20251001_20260728.csv" \
  --time_col time \
  --target_col value \
  --freq D \
  --aggregation_enabled true \
  --aggregation_source_freq 5min \
  --aggregation_method mean \
  --aggregation_fill_method seasonal_slot \
  --aggregation_fill_weeks 4 \
  --aggregation_output_path "dataset/aidc_power_month/derived/${ROUTE}_Loads_1day_mean_20251001_20260728.csv" \
  --model_name "$model_name" \
  --model_params '{}' \
  --forecast_strategy direct \
  --do_train false \
  --do_test false \
  --do_forecast false \
  --do_eda true \
  --eda_period 7 \
  --eda_nlags 24 \
  --eda_run_preprocessed false \
  --eda_recommendation_enabled true \
  --history_size 150 \
  --predict_horizon 30 \
  --results_data_name "aidc_power_month/route_${ROUTE}" \
  --results_dir results
