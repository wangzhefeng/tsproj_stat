# 监控闭环

本地文件版监控，不依赖数据库或服务端组件。

## 预测日志

`monitor_enabled=true` 时，forecast 阶段把每个未来步的 `yhat` 写入：

```text
results/{data_name}/monitor/{experiment_path}/predictions_log.csv
```

启用示例：

```bash
.venv/bin/python run.py --model_name naive \
  --do_train true --do_test true --do_forecast true \
  --backtest_n_jobs 2 --monitor_enabled true
```

## 实际值回填

`evaluation.monitor.ModelMonitor` 或 `run.py --monitor_actuals_path ...` 支持后续回填真实值到 `actuals_log.csv`，并基于最近 `monitor_window` 个匹配样本生成 `metrics_history.csv`：

```bash
.venv/bin/python run.py \
  --monitor_actuals_path /abs/path/actuals.csv \
  --monitor_actuals_experiment_path naive-direct/params-default/... \
  --monitor_actuals_value_col actual \
  --monitor_actuals_run_id manual-backfill-1
```

- 回填入口已收口到 `evaluation.monitor.run_monitor_actuals_backfill()`；`run.py` 只负责解析配置并调用统一入口
- 面板入口不与监控回填混用（见 [exogenous.md](exogenous.md#面板多序列多模型)）
