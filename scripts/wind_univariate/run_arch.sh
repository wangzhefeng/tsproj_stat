#!/usr/bin/env bash

set -euo pipefail

# 单变量风电脚本：使用 dataset/wind_dataset.csv，DATE 为时间列，WIND 为目标列。
# 该脚本显式传入完整 AppConfig 字段，便于复现实验参数和结果目录。
cd "$(dirname "$0")/../.."

model_name=arch
export LOG_NAME="$model_name"

# 运行完整主流程：训练、rolling backtest 和未来预测，结果写入 saved_results/{setting}/。
python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/wind_dataset.csv \
  --time_col DATE \
  --target_col WIND \
  --freq D \
  --model_name "$model_name" \
  --model_params '{}' \
  --inference_strategy direct \
  --do_train true \
  --do_test true \
  --do_forecast true \
  --do_eda true \
  --history_size 365 \
  --predict_horizon 7 \
  --backtest_train_size 365 \
  --backtest_horizon 7 \
  --backtest_step 7 \
  --backtest_window_mode expanding \
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
  --forecast_result_dir saved_results/results_forecast \
  --eda_output_dir saved_results/results_eda
