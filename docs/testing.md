# 回测与验证

## 滚动回测

`do_test` 阶段承担历史评估（rolling backtest）：

- 窗口模式 `backtest_window_mode = expanding | sliding`；`backtest_train_size` 控制训练窗长
- 不变量：`backtest_train_size` 未显式设置时默认等于 `history_size`（回测与 final fit 同窗口）；解析优先级 `backtest_train_size` > 旧字段 `backtest_initial_train_size` > `history_size`
- `backtest_n_jobs > 1` 按窗口并行，CSV 输出仍按 `window_id` 稳定排序
- 失败策略：回测窗口失败默认 RAISE；容忍须显式 `--backtest_allow_failed_windows true` 且产物打标 `survivor_bias`
- forecast 输出 NaN 默认 RAISE；`--forecast_allow_nan_fill true` 容忍并打标 `forecast_nan_filled`
- 输出：窗口级明细、汇总指标、三类图（预测对比、残差、误差分布）；区间指标多水平时按水平展开为 `interval_coverage_80` 式带后缀列（见 [strategies.md](strategies.md#策略与区间)）
- 重拟合调度 `backtest_refit_every` 见 [strategies.md](strategies.md#重拟合与状态更新)
- 训练修复只看各自历史窗口；回测/校准真值不插值，缺失即失败。点回测记录 `history_filled_value_count`；区间回测标记 `per_calibration_origin`，不冒充一个预填充窗口。离线派生输入的双向填充不受此保证覆盖，须检查聚合审计。

## 验证命令基线

```bash
# 默认基线
.venv/bin/python -m pytest -q

# CLI 烟雾（完整清单见 AGENTS.md §3，此处为代表性子集）
.venv/bin/python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5
.venv/bin/python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5 --backtest_n_jobs 2 --monitor_enabled true
.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false
bash scripts/wind_univariate/run_eda.sh
bash scripts/aidc_power_month/route_A/run_eda.sh
bash scripts/aidc_power_month/route_B/run_eda.sh
```

MC/Hermes 会话统一加 `env -u PYTHONPATH` 前缀。

## 测试内容约定

- 算法测试验证数值或行为，不以输出长度、类名或路径键代替正确性断言
- 覆盖：去噪数值、趋势与季节相位还原、MAPE/SMAPE 比例与零分母口径、预测 CSV 数值与时间轴、NeuralProphet 依赖失败路径显式注入
- LinearVAR 回归覆盖 lag 0/1、显式未来输入覆盖、未来行数不足报错、train/test/forecast 与归档模型数值一致性
- 数据层边界覆盖内存聚合数值/输入不变性、CSV/内存清洗一致性、AIDC 入口跨 cwd 执行及旧缓存审计补齐；重构数值对照见 [数据层重构记录](data-provider-refactor.md)
- `test_data_boundaries.py` 覆盖修复前质检、EDA 不隐式改数、窗口/校准修复边界、业务尺度预测/回测/模拟/拟合值、变换器归档及未来外生时间对齐。
- `tests/` 纳入版本控制，与生产代码一同审核；只忽略测试缓存

完整 CLI smoke 清单与依赖锁定说明见 [setup.md](setup.md) 与 AGENTS.md §3；实施验收记录见 [statsforecast-extension.md](statsforecast-extension.md)。
