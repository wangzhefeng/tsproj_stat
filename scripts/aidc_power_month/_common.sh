#!/usr/bin/env bash
# AIDC 月均功率场景公共运行体：route_A/route_B 的单模型变体脚本仅差数据路径与结果子树，
# 差异由 ROUTE 派生；变体参数（模型/超参/去趋势/回测进度）由 variants/<name>.sh 设置。
# 变体脚本协议：先 export ROUTE=A|B，再设置 MODEL_NAME，可选覆盖
# MODEL_PARAMS / DETREND_METHOD / BACKTEST_VERBOSE / LOG_NAME_OPT，最后调用 run_single_model。
set -euo pipefail
: "${ROUTE:?ROUTE must be A or B}"

DATA_PATH="dataset/aidc_power_month/${ROUTE}_Loads_5min_20251001_20260728.csv"
DERIVED_PATH="dataset/aidc_power_month/derived/${ROUTE}_Loads_1day_observed_20251001_20260728.csv"
RESULTS_DATA_NAME="aidc_power_month/route_${ROUTE}"

run_single_model() {
  local model_name="${MODEL_NAME:?MODEL_NAME required}"
  local model_params="${MODEL_PARAMS:-'{}'}"
  local detrend_method="${DETREND_METHOD:-none}"
  local backtest_verbose="${BACKTEST_VERBOSE:-false}"
  export LOG_NAME="${LOG_NAME_OPT:-${MODEL_NAME}}"

  local common_dir
  common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${common_dir}/../.."

  # 运行完整主流程：训练、rolling backtest 和未来预测。
  # 脚本入口统一为 .venv/bin/python -u run.py（项目根 .venv 直调）。
  .venv/bin/python -u run.py \
    --project_name tsproj_stat \
    --seed 2026 \
    --data_path "${DATA_PATH}" \
    --time_col time \
    --target_col value \
    --freq D \
    --aggregation_enabled true \
    --aggregation_source_freq 5min \
    --aggregation_method mean \
    --aggregation_fill_method preserve \
    --require_aggregation_audit true \
    --backtest_missing_target_policy exclude \
    --aggregation_fill_weeks 4 \
    --aggregation_output_path "${DERIVED_PATH}" \
    --model_name "${model_name}" \
    --model_params "${model_params}" \
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
    --backtest_verbose "${backtest_verbose}" \
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
    --detrend_method "${detrend_method}" \
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
    --results_data_name "${RESULTS_DATA_NAME}" \
    --results_dir results
}
