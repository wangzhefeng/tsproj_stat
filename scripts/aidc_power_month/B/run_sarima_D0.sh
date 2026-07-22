#!/usr/bin/env bash

set -euo pipefail

# AIDC 负荷（B路，日峰）单变量脚本：从 dataset/aidc_power_month/B_Loads_5min_20251001_20260708.csv 聚合生成 derived/B_Loads_1day_20251001_20260708.csv，time 为时间列，value 为目标列。
# SARIMA 季节差分关闭(D=0)：seasonal_order=[1,0,1,7]，对应 EDA 的 D=0 建议；保留周季节 AR/MA 项但不做季节差分，避免弱季节性下过度差分。
cd "$(dirname "$0")/../../.."

model_name=sarima
export LOG_NAME="sarima_D0"

# 运行完整主流程：训练、rolling backtest 和未来预测。
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
  --model_params '{"order":[1,1,1],"seasonal_order":[1,0,1,7],"enforce_stationarity":false,"enforce_invertibility":false,"fit_kwargs":{"disp":false,"maxiter":20}}' \
  --forecast_strategy direct \
  --do_train true \
  --do_test true \
  --do_forecast true \
  --do_eda false \
  --history_size 150 \
  --predict_horizon 30 \
  --backtest_train_size 150 \
  --backtest_horizon 30 \
  --backtest_step 30 \
  --backtest_window_mode sliding \
  --backtest_verbose true \
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
