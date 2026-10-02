# 策略与区间

## 预测策略

`forecast_strategy`：`native / single_step / direct / recursive / dirrec`，默认 `direct`。

- `native`：一次拟合原生多步（点预测与区间共用模型状态）
- 旧 `direct`：同历史逐步重拟合并取对应步长末值，是兼容语义，不等同于监督学习 horizon-specific direct
- `recursive / dirrec`：追加预测后逐步重拟合，不冒充参数固定的在线更新
- 点预测 NaN 默认 RAISE；仅显式 `forecast_allow_nan_fill=true` 才容忍并在产物标记 `forecast_nan_filled`。

## 原生多步、区间与多季节分解

```bash
env -u PYTHONPATH .venv/bin/python run.py --model_name auto_ets --forecast_strategy native --return_intervals true --history_size 60 --predict_horizon 5
env -u PYTHONPATH .venv/bin/python run.py --model_name naive --forecast_strategy recursive --return_intervals true --interval_method conformal --interval_alpha 0.2 --conformal_n_windows 4 --history_size 60 --predict_horizon 5
env -u PYTHONPATH .venv/bin/python run.py --model_name historic_average --forecast_strategy native --decomposition_method mstl --seasonal_periods 7,24 --history_size 180 --backtest_train_size 180 --predict_horizon 5
```

- `interval_method=native`：调用模型区间；要求模型明确支持且边界有限、有序；`recursive / dirrec` 下显式 RAISE 并提示改用 conformal（P6，不再产出 NaN 区间列）
- `interval_method=conformal`：按实际策略逐步长绝对误差校准；每个校准窗口重新拟合 DataProcessor，误差与区间都在原始尺度。校准段互不重叠、训练前缀扩展；需 `history_size >= conformal_n_windows * predict_horizon + 3`；有限样本顺序统计量不可达时显式失败，不静默夹紧置信水平。默认 20 窗；95% 水平至少 19 窗
- 多置信水平（P7）：`--interval_levels 0.8 0.95`（小数列表）一次产出多水平区间；缺省回退单水平 `[1-interval_alpha]`，行为与旧版完全一致。单水平列名保持 `yhat_lower/yhat_upper`；多水平列为 `yhat_lower_80 / yhat_upper_95` 式带后缀。conformal 多水平共享同一次校准循环（拟合成本不随水平数增加）；StatsForecast 后端一次 `predict(level=[...])` 返回全部水平。任一水平有限样本不可达即整体失败，不静默丢弃该水平
- 启用区间后回测输出 `interval_coverage / interval_width / winkler_score` 及逐点上下界；多水平时按水平展开为 `interval_coverage_80` 式带后缀列。覆盖率是经验评估，不保证非平稳序列的名义覆盖
- Winkler 使用实际有效置信水平；单水平 `interval_levels` 显式覆盖时，不再使用默认 alpha 评分。
- MSTL 最大周期必须严格小于各训练/校准窗口长度一半，短序列显式失败
- 当前区间路径要求 `scale=false`、`feature_mode=analysis_snapshot`（见 [limitations.md](../pipeline/limitations.md)）

## 样本路径模拟（P9）

```bash
env -u PYTHONPATH .venv/bin/python run.py --model_name naive --forecast_strategy recursive \
  --simulate_enabled true --simulate_n_paths 100 --simulate_error_distribution bootstrap \
  --simulate_n_windows 8 --history_size 40 --predict_horizon 4 --do_forecast true --do_train false --do_test false
```

- 误差驱动路径集成：滚动起点带符号误差池（与 conformal 同源、原始尺度，共享 `forecasting/origins.py` 的 rolling_error_pool）驱动路径抽样；任意模型 × 策略通用
- `--simulate_error_distribution bootstrap|normal|t|laplace`：bootstrap=整窗行向量有放回抽样（保留步长间相关）；normal=逐步高斯；t/laplace=逐 step 从误差池 scipy MLE 拟合分布参数后抽样（需 >=10 个校准窗，t 的拟合 df<=2 时显式拒绝）
- 产物独立：`simulated_paths.csv`（path_id/step/value 长表）与 `simulated_quantiles.csv`（q10 式分位列），不与 forecast.csv 区间列混排
- 可与区间同时开启；两条计算路径独立执行并验产物。派生特征模式下模拟校准也按窗口构造特征，见 [features.md](../features/features.md)。
- seed 沿用 `--seed`；时间相关与分布漂移下不承诺无条件路径分布保证，校准样本不足显式失败

## 重拟合与状态更新

```bash
env -u PYTHONPATH .venv/bin/python run.py --model_name arima --model_params '{"order":[1,0,0]}' --forecast_strategy native --backtest_refit_every 2 --backtest_n_jobs 1
```

`backtest_refit_every=1` 每窗拟合（默认），N 每 N 窗拟合，0 仅首窗拟合；中间窗口用固定参数在当前真实历史重新滤波（不是重新估计）。仅 AR/MA/ARMA/ARIMA/SARIMA 支持；要求 native、串行、无可逆预处理、无区间，其它组合明确拒绝。`refit_count` 统计成功重拟合窗口数。

相关：[组件契约](contracts.md)（区间裁决、共享误差池、`forecast_use_update` 前向快速路径）、[testing.md](../evaluation/testing.md)（回测窗口）、[preprocessing.md](../data_provider/preprocessing.md)（分解）。
