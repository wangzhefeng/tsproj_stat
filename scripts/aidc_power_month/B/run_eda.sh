#!/usr/bin/env bash

set -euo pipefail

# AIDC B 路日峰数据项目级 EDA：先生成或复用日频派生数据，再独立执行 EDA。
cd "$(dirname "$0")/../../.."

model_name=naive
export LOG_NAME=eda_B_Loads_1day

python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/aidc_power_month/B_Loads_5min_20251001_20260708.csv \
  --time_col time \
  --target_col value \
  --freq D \
  --aggregation_enabled true \
  --aggregation_source_freq 5min \
  --aggregation_method max \
  --aggregation_fill_method seasonal_slot \
  --aggregation_fill_weeks 4 \
  --aggregation_output_path dataset/aidc_power_month/derived/B_Loads_1day_20251001_20260708.csv \
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
  --history_size 120 \
  --predict_horizon 30 \
  --results_dir results
