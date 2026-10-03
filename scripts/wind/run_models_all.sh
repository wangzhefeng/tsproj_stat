#!/usr/bin/env bash

set -euo pipefail
# 场景级多模型合并脚本：一次 run 完成 25 个单变量基座模型对比（替代逐模型单 shell）。
# 每模型超参见 --batch_models；neuralprophet 因环境损坏未纳入（原脚本保留）；
# 多变量模型（var/bayesian_var/linear_var）由 multivariate/run_models_all.sh 单独运行。
# 数据加载/聚合/预处理只做一次；各模型独立 experiment_path。
# do_test=true：产出横向对比表 results_{data_name}/results_test/comparison/model_comparison.csv；
# 注意 25 模型 × expanding 回测（step=7，约 887 个原点）耗时较长，auto_arima/sarima 为主成本。


cd "$(dirname "$0")/../.."

export LOG_NAME="multi_model"

# 运行完整主流程：训练、rolling backtest 和未来预测。
# 脚本入口统一为 .venv/bin/python -u run.py（项目根 .venv 直调）。
.venv/bin/python -u run.py \
  --project_name tsproj_stat \
  --seed 2026 \
  --data_path dataset/wind/wind_dataset.csv \
  --time_col DATE \
  --target_col WIND \
  --freq D \
  --model_names ar,arch,arima,arma,auto_arima,auto_ces,auto_ets,auto_theta,bayesian_tmt,croston,dynamic_theta,ets,garch,historic_average,ma,naive,prophet,rar,random_walk_drift,sarima,seasonal_naive,seasonal_window_average,sf_auto_arima,tbats,theta \
  --batch_models '{"ar": {"p": 2}, "arch": {}, "arima": {"order": [1, 1, 1]}, "arma": {"p": 1, "q": 1}, "auto_arima": {"seasonal": false, "m": 1, "stepwise": true, "start_p": 0, "start_q": 0, "max_p": 2, "max_q": 2, "max_order": 4, "maxiter": 20, "information_criterion": "aic", "trace": true, "error_action": "ignore", "suppress_warnings": true}, "auto_ces": {"season_length": 7}, "auto_ets": {"season_length": 7, "freq": "D"}, "auto_theta": {"season_length": 7, "freq": "D"}, "bayesian_tmt": {"lags": [1, 2, 7]}, "croston": {"alpha": 0.2}, "dynamic_theta": {"season_length": 7, "freq": "D"}, "ets": {"trend": "add", "seasonal": "add", "seasonal_periods": 7}, "garch": {}, "historic_average": {"window": 365}, "ma": {"q": 1}, "naive": {}, "prophet": {"freq": "D"}, "rar": {"alpha": 0.2}, "random_walk_drift": {}, "sarima": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 7], "enforce_stationarity": false, "enforce_invertibility": false, "simple_differencing": true, "fit_kwargs": {"disp": false, "maxiter": 20}}, "seasonal_naive": {"season_length": 7}, "seasonal_window_average": {"season_length": 7, "window_size": 2}, "sf_auto_arima": {}, "tbats": {"seasonal_periods": [7], "show_warnings": false, "n_jobs": 1}, "theta": {"period": 7}}' \
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
