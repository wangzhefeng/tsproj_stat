#!/usr/bin/env bash

set -euo pipefail
# 场景级多模型合并脚本：一次 run 完成 3 个多变量模型对比（var / bayesian_var / linear_var）。
# 内生面板 = WIND + RAIN/T.MAX/T.MIN/T.MIN.G；缺失窗口内线性插值修复并逐列入审计；评估只看 WIND。
# 数据加载/预处理只做一次；各模型独立 experiment_path。
# do_test=true：产出横向对比表 results_wind_dataset/results_test/comparison/model_comparison.csv。
# 注意：为控制 bayesian_var 抽样成本，批量统一 backtest_step=30；var/linear_var 单模型脚本用 step=7，
# 批量表与单脚本结果网格不同，精确对比以单模型脚本为准。

cd "$(dirname "$0")/../../.."

export LOG_NAME="multi_model_wind_multivariate"

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
  --model_names var,bayesian_var,linear_var \
  --batch_models '{"var": {"maxlags": 14, "ic": "aic"}, "bayesian_var": {}, "linear_var": {}}' \
  --forecast_strategy native \
  --do_train true \
  --do_test true \
  --do_forecast true \
  --do_eda false \
  --history_size 365 \
  --predict_horizon 7 \
  --backtest_train_size 365 \
  --backtest_horizon 7 \
  --backtest_step 30 \
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
