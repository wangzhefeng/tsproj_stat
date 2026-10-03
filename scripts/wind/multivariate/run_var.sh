#!/usr/bin/env bash

set -euo pipefail

# 多变量风电脚本：dataset/wind/wind_dataset.csv，DATE 为时间列，WIND 为目标列。
# 内生面板 = WIND + RAIN/T.MAX/T.MIN/T.MIN.G；IND* 为缺失标记列，不入面板。
# T.MAX/T.MIN/T.MIN.G 的大段缺失由窗口内线性插值修复（data_provider/cleaning/imputation.py，
# policy=window_local_linear，逐列填补量写入 run 审计 history_filled_by_column）；评估口径只看 WIND。
# VAR（statsmodels 后端）联合建模 5 列内生面板，预测只抽取 WIND；maxlags=14 + aic 选阶（覆盖两周周期）。
cd "$(dirname "$0")/../../.."

model_name=var
export LOG_NAME="$model_name"

# 运行完整主流程：训练、rolling backtest 和未来预测。
# 脚本入口统一为 .venv/bin/python -u run.py（项目根 .venv 直调）。
.venv/bin/python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/wind/wind_dataset.csv \
  --time_col DATE \
  --target_col WIND \
  --endog_cols RAIN,T.MAX,T.MIN,T.MIN.G \
  --freq D \
  --model_name "$model_name" \
  --model_params '{"maxlags":14,"ic":"aic"}' \
  --forecast_strategy native \
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
