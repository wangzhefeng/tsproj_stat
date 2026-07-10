#!/usr/bin/env bash

set -euo pipefail

# 风电数据项目级 EDA：每份数据单独运行一次，不执行模型训练、回测或预测。
cd "$(dirname "$0")/../.."

model_name=naive
export LOG_NAME=eda_wind_dataset

python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/wind_dataset.csv \
  --time_col DATE \
  --target_col WIND \
  --freq D \
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
  --history_size 365 \
  --predict_horizon 7 \
  --results_dir results
