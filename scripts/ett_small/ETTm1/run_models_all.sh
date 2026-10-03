#!/usr/bin/env bash

set -euo pipefail

# ETTm1 单变量批量：与 univariate/ 单模型入口共用实验口径。
# neuralprophet 仅独立运行；多变量批量见 multivariate/run_models_all.sh。
# 完整 expanding 回测较慢，先用独立模型和隔离结果验证。
cd "$(dirname "$0")/../../.."
export LOG_NAME=multi_model_ETTm1_univariate

.venv/bin/python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/ETT-small/ETTm1.csv \
  --time_col date \
  --target_col OT \
  --freq 15min \
  --results_data_name ett_small/ETTm1 \
  --model_names ar,arch,arima,arma,auto_arima,auto_ces,auto_ets,auto_theta,bayesian_tmt,croston,dynamic_theta,ets,garch,historic_average,ma,naive,prophet,rar,random_walk_drift,sarima,seasonal_naive,seasonal_window_average,sf_auto_arima,tbats,theta \
  --batch_models '{"ar": {"p": 2}, "arch": {}, "arima": {"order": [1, 1, 1]}, "arma": {"p": 1, "q": 1}, "auto_arima": {"seasonal": false, "m": 1, "stepwise": true, "start_p": 0, "start_q": 0, "max_p": 2, "max_q": 2, "max_order": 4, "maxiter": 20, "information_criterion": "aic", "trace": true, "error_action": "ignore", "suppress_warnings": true}, "auto_ces": {"season_length": 96}, "auto_ets": {"season_length": 96, "freq": "15min"}, "auto_theta": {"season_length": 96, "freq": "15min"}, "bayesian_tmt": {"lags": [1, 2, 96]}, "croston": {"alpha": 0.2}, "dynamic_theta": {"season_length": 96, "freq": "15min"}, "ets": {"trend": "add", "seasonal": "add", "seasonal_periods": 96}, "garch": {}, "historic_average": {"window": 1920}, "ma": {"q": 1}, "naive": {}, "prophet": {"freq": "15min"}, "rar": {"alpha": 0.2}, "random_walk_drift": {}, "sarima": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 96], "enforce_stationarity": false, "enforce_invertibility": false, "simple_differencing": true, "fit_kwargs": {"disp": false, "maxiter": 20}}, "seasonal_naive": {"season_length": 96}, "seasonal_window_average": {"season_length": 96, "window_size": 2}, "sf_auto_arima": {}, "tbats": {"seasonal_periods": [96], "show_warnings": false, "n_jobs": 1}, "theta": {"period": 96}}' \
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
