#!/usr/bin/env bash

set -euo pipefail

# ETTm1 单变量 OT 基线；15min，日周期 96；不读取协变量。
# 训练/回测/预测窗口与多变量场景一致；CLI 参数可显式覆盖。
cd "$(dirname "$0")/../../../.."

model_name=arch
export LOG_NAME="$model_name"

.venv/bin/python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/ETT-small/ETTm1.csv \
  --time_col date \
  --target_col OT \
  --freq 15min \
  --results_data_name ett_small/ETTm1 \
  --model_name "$model_name" \
  --model_params '{}' \
  --forecast_strategy native \
  --do_train true \
  --do_test true \
  --do_forecast true \
  --do_eda false \
  --history_size 1920 \
  --predict_horizon 96 \
  --backtest_train_size 1920 \
  --backtest_horizon 96 \
  --backtest_step 480 \
  --backtest_window_mode expanding \
  --backtest_verbose false \
  --backtest_progress_every 10 \
  --backtest_n_jobs 1 \
  --feature_mode analysis_snapshot \
  --enable_datetime_features true \
  --lags 1,2,96,192 \
  --scale false \
  --scaler_type standard \
  --denoise_enabled false \
  --denoise_method none \
  --denoise_window 3 \
  --detrend_method none \
  --seasonal_period 96 \
  --decomposition_method none \
  --decomposition_target trend_resid \
  --decomposition_model additive \
  --acf_max_lag 192 \
  --seasonality_strength_threshold 0.3 \
  --ets_tune_smoothing_params false \
  --auto_select false \
  --auto_select_metric mae \
  --auto_select_n_windows 5 \
  --max_missing_ratio 0.3 \
  --validate_freq true \
  --return_intervals false \
  --interval_alpha 0.05 \
  --monitor_enabled false \
  --monitor_window 30 \
  --log_format text \
  --results_dir results \
  "$@"
