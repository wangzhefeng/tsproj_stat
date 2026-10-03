# CLI 烟雾验证

按本次改动选择相关场景；这些命令不是每次编辑都必须运行的清单。全量/定向判定见 [验证约定](README.md)。

## 无业务数据的基础入口

从项目根运行；下列命令使用内置 demo，实际会生成产物。隔离验收时追加 `--results_dir` 指定临时绝对路径。

```bash
.venv/bin/python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5
.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false
.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false --eda_period 7 --eda_nlags 24 --eda_recommendation_enabled true
.venv/bin/python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5 --backtest_n_jobs 2 --monitor_enabled true
```

## 需要本地数据的场景

```bash
bash scripts/wind/eda/run_eda.sh
bash scripts/aidc_power_month/route_A/eda/run_eda.sh
bash scripts/aidc_power_month/route_B/eda/run_eda.sh
bash scripts/ett_small/ETTm1/eda/run_eda.sh
```

- 先检查业务数据和聚合审计；这类脚本写正式结果，还可能重建派生输入，不为文档检查而运行。
- MC/Hermes 会话的 Python 命令前加 `env -u PYTHONPATH`；shell 入口可用 `env -u PYTHONPATH bash ...` 清除继承变量。

## 外生变量与监控回填模板

以下是模板，不可原样运行；替换绝对路径、有效实验路径及所需列，先准备真实输入和已有监控预测。

```bash
.venv/bin/python run.py --monitor_actuals_path /abs/path/actuals.csv --monitor_actuals_experiment_path naive-direct/params-default/... --monitor_actuals_value_col actual --monitor_actuals_run_id manual-backfill-1
.venv/bin/python run.py --data_path /abs/path/history.csv --time_col ds --target_col y --model_name linear_var --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' --endog_cols load --exog_cols temp --future_exog_path /abs/path/future_exog.csv --future_exog_time_col ds --future_exog_cols temp --do_train true --do_test true --do_forecast true --history_size 12 --predict_horizon 4
```

输入契约见 [外生与面板](../pipeline/exogenous.md)；实际值匹配和重复回填拒绝规则见 [监控](../monitoring/monitoring.md)。
