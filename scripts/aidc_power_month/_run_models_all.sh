#!/usr/bin/env bash
# 场景级多模型合并共用体：一次 run 完成 21 个基座模型对比（替代逐模型单 shell）。
# 每模型超参见 --batch_models；本清单未选择 neuralprophet，其独立入口保留。
# 参数变体对照（arima_110/210、ets_trend、sarima_D0、theta_p1、ar_detrend_*）参数轴不同，保留独立脚本。
# 数据加载/聚合/预处理共用，EDA 关闭；各模型产物隔离。
# 对比表：results/<data_name>/results_test/comparison/runs/<run_id>/model_comparison.csv。
# 由 route_A/route_B/run_models_all.sh 设置 ROUTE 后 exec。
set -euo pipefail
: "${ROUTE:?ROUTE must be A or B}"

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${_script_dir}/../.."

export LOG_NAME="multi_model"

# 运行完整主流程：训练、rolling backtest 和未来预测。
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
  --model_names ar,arch,arima,arma,auto_arima,auto_ets,auto_theta,bayesian_tmt,croston,dynamic_theta,ets,garch,historic_average,ma,naive,prophet,rar,sarima,seasonal_naive,tbats,theta \
  --batch_models '{"ar": {"p": 2}, "arch": {}, "arima": {"order": [1, 1, 1]}, "arma": {"p": 1, "q": 1}, "auto_arima": {"seasonal": false, "m": 1, "stepwise": true, "start_p": 0, "start_q": 0, "max_p": 2, "max_q": 2, "max_order": 4, "maxiter": 20, "information_criterion": "aic", "trace": true, "error_action": "ignore", "suppress_warnings": true}, "auto_ets": {"season_length": 7, "freq": "D"}, "auto_theta": {"season_length": 7, "freq": "D"}, "bayesian_tmt": {"lags": [1, 2, 7]}, "croston": {"alpha": 0.2}, "dynamic_theta": {"season_length": 7, "freq": "D"}, "ets": {"trend": "add", "seasonal": "add", "seasonal_periods": 7}, "garch": {}, "historic_average": {"window": 180}, "ma": {"q": 1}, "naive": {}, "prophet": {"freq": "D"}, "rar": {"alpha": 0.2}, "sarima": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7], "enforce_stationarity": false, "enforce_invertibility": false, "fit_kwargs": {"disp": false, "maxiter": 20}}, "seasonal_naive": {"season_length": 7}, "tbats": {"seasonal_periods": [7], "show_warnings": false, "n_jobs": 1}, "theta": {"period": 7}}' \
  --forecast_strategy native \
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
  --results_data_name "aidc_power_month/route_${ROUTE}" \
  --results_dir results
