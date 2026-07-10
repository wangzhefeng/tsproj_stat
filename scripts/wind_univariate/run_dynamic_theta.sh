#!/usr/bin/env bash

set -euo pipefail

# 单变量风电脚本：使用 dataset/wind_dataset.csv，DATE 为时间列，WIND 为目标列。
# dynamic_theta 属于 optional StatsForecast 模型，显式传入日频和周季节长度。
cd "$(dirname "$0")/../.."

model_name=dynamic_theta
export LOG_NAME="$model_name"

# 运行完整主流程：训练、rolling backtest 和未来预测。
# 脚本入口按 wind_univariate 约定保持 python -u run.py。
python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/wind_dataset.csv \
  --time_col DATE \
  --target_col WIND \
  --freq D \
  --model_name "$model_name" \
  --model_params '{"season_length":7,"freq":"D"}' \
  --forecast_strategy direct \
  --do_train true \
  --do_test true \
  --do_forecast true \
  --do_eda false \
  --history_size 365 \
  --predict_horizon 7 \
  --backtest_train_size 365 \
  --backtest_horizon 7 \
  --backtest_step 7 \
  --backtest_window_mode expanding \
  --backtest_verbose false \
  --backtest_progress_every 10 \
  --backtest_n_jobs 1 \
  --feature_mode analysis_snapshot \
  --enable_datetime_features true \
  --lags 1,2,7,14 \
  --scale false \
  --scaler_type standard \
  --denoise_enabled false \
  --denoise_method none \
  --denoise_window 3 \
  --detrend_method none \
  --seasonal_period 7 \
  --decomposition_method none \
  --decomposition_target trend_resid \
  --decomposition_model additive \
  --acf_max_lag 48 \
  --seasonality_strength_threshold 0.3 \
  --ets_tune_smoothing_params false \
  --auto_select false \
  --auto_select_candidates naive,seasonal_naive,historic_average,arima,auto_arima,ets,theta \
  --auto_select_metric mae \
  --auto_select_n_windows 5 \
  --max_missing_ratio 0.3 \
  --validate_freq true \
  --return_intervals false \
  --interval_alpha 0.05 \
  --monitor_enabled false \
  --monitor_window 30 \
  --log_format text \
  --results_dir results
