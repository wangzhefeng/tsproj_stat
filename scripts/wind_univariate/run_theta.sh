#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

model_name=theta
export LOG_NAME="$model_name"

python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/wind_dataset.csv \
  --time_col DATE \
  --target_col WIND \
  --freq D \
  --model_name "$model_name" \
  --model_params '{}' \
  --pred_method direct \
  --do_train true \
  --do_test true \
  --do_forecast true \
  --do_eda false \
  --history_size 365 \
  --predict_horizon 7 \
  --backtest_initial_train_size 365 \
  --backtest_horizon 7 \
  --backtest_step 7 \
  --enable_datetime_features true \
  --lags 1,2,7,14 \
  --scale false \
  --scaler_type standard \
  --denoise_enabled false \
  --denoise_window 3 \
  --detrend_method none \
  --checkpoints_dir saved_results/checkpoints \
  --train_results_dir saved_results/results_train \
  --test_results_dir saved_results/results_test \
  --pred_results_dir saved_results/results_forecast \
  --eda_output_dir saved_results/results_eda
