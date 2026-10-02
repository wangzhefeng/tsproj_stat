# 监控闭环

本地文件版监控，不依赖数据库或服务端组件。
监控保持实验级累计，不加 `runs/`；v2 实验身份与输入内容指纹分离（见 [artifacts.md](../artifacts/artifacts.md)）。旧实验仍可按显式原路径回填。

## 预测日志

`monitor_enabled=true` 时，forecast 阶段把每个未来步的 `yhat` 与目标时间戳 `target_ts` 写入：

```text
results/{data_name}/monitor/{experiment_path}/predictions_log.csv
```

启用示例：

```bash
.venv/bin/python run.py --model_name naive \
  --do_train true --do_test true --do_forecast true \
  --backtest_n_jobs 2 --monitor_enabled true
```

## 区间覆盖率跟踪（P7）

预测含区间列时（单水平或 `--interval_levels` 多水平），predictions_log.csv 同步记录区间上下界列；实际值回填后，滚动指标除 MAE/RMSE/MAPE 外还输出区间覆盖率：

- 单水平：`interval_coverage`
- 多水平：逐水平 `interval_coverage_80 / interval_coverage_95` 式带后缀（与区间列名同后缀）
- 无区间列或区间值缺失时不产出覆盖率键，不伪造数值；旧 CSV 表头缺失的水平列在写入时自动迁移补列

```bash
.venv/bin/python run.py --model_name sf_auto_arima --forecast_strategy native \
  --return_intervals true --interval_levels 0.8 0.95 \
  --do_train true --do_test true --do_forecast true \
  --monitor_enabled true
```

覆盖率随实际值逐批回填滚动更新，用于发现区间校准漂移（名义 80% 经验覆盖持续偏低即区间过窄）。

## 实际值回填

`monitoring.monitor.ModelMonitor` 或 `run.py --monitor_actuals_path ...` 支持后续回填真实值到 `actuals_log.csv`，并基于最近 `monitor_window` 个匹配样本生成 `metrics_history.csv`：

```bash
.venv/bin/python run.py \
  --monitor_actuals_path /abs/path/actuals.csv \
  --monitor_actuals_experiment_path naive-direct/params-default/... \
  --monitor_actuals_value_col actual \
  --monitor_actuals_run_id manual-backfill-1
```

- 回填入口已收口到 `monitoring.monitor.run_monitor_actuals_backfill()`；`run.py` 只负责解析配置并调用统一入口
- 回填批内、批间统一判重；重复或旧双键与新三键歧义重叠均 RAISE，拒绝前不追加任何行。时间键规范化比较（Z 后缀/空格书写差异不影响判重）。
- 按记录匹配：双方 target_ts 非空须三键相同；一方为空才尝试双键。一个记录对应多个候选时 RAISE，不重复计权；新旧日志混用不丢失先前已匹配样本。
- 当前为单写者本地 CSV 接口；判重与追加未加跨进程事务锁，不承诺并发回填安全。
- 滚动点指标由 `evaluation.metrics.POINT_METRICS` 注册表派生（mae/rmse/mape/smape/mse/r2/bias/max_error），非有限值在快照表写空串
- 面板入口不与监控回填混用（见 [exogenous.md](../pipeline/exogenous.md#面板多序列多模型)）
