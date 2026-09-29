# 策略与区间

## 预测策略

`forecast_strategy`：`native / single_step / direct / recursive / dirrec`，默认 `direct`。

- `native`：一次拟合原生多步（点预测与区间共用模型状态）
- 旧 `direct`：同历史逐步重拟合并取对应步长末值，是兼容语义，不等同于监督学习 horizon-specific direct
- `recursive / dirrec`：追加预测后逐步重拟合，不冒充参数固定的在线更新

## 原生多步、区间与多季节分解

```bash
env -u PYTHONPATH .venv/bin/python run.py --model_name auto_ets --forecast_strategy native --return_intervals true --history_size 60 --predict_horizon 5
env -u PYTHONPATH .venv/bin/python run.py --model_name naive --forecast_strategy recursive --return_intervals true --interval_method conformal --interval_alpha 0.2 --conformal_n_windows 4 --history_size 60 --predict_horizon 5
env -u PYTHONPATH .venv/bin/python run.py --model_name historic_average --forecast_strategy native --decomposition_method mstl --seasonal_periods 7,24 --history_size 180 --backtest_train_size 180 --predict_horizon 5
```

- `interval_method=native`：调用模型区间；要求模型明确支持且边界有限、有序；`recursive / dirrec` 下仍是 NaN（需区间时用 conformal）
- `interval_method=conformal`：按实际策略逐步长绝对误差校准；每个校准窗口重新拟合 DataProcessor，误差与区间都在原始尺度。校准段互不重叠、训练前缀扩展；需 `history_size >= conformal_n_windows * predict_horizon + 3`；有限样本顺序统计量不可达时显式失败，不静默夹紧置信水平。默认 20 窗；95% 水平至少 19 窗
- 启用区间后回测输出 `interval_coverage / interval_width / winkler_score` 及逐点上下界；覆盖率是经验评估，不保证非平稳序列的名义覆盖
- MSTL 最大周期必须严格小于各训练/校准窗口长度一半，短序列显式失败
- 当前区间路径要求 `scale=false`、`feature_mode=analysis_snapshot`（见 [limitations.md](limitations.md)）

## 重拟合与状态更新

```bash
env -u PYTHONPATH .venv/bin/python run.py --model_name arima --model_params '{"order":[1,0,0]}' --forecast_strategy native --backtest_refit_every 2 --backtest_n_jobs 1
```

`backtest_refit_every=1` 每窗拟合（默认），N 每 N 窗拟合，0 仅首窗拟合；中间窗口用固定参数在当前真实历史重新滤波（不是重新估计）。仅 AR/MA/ARMA/ARIMA/SARIMA 支持；要求 native、串行、无可逆预处理、无区间，其它组合明确拒绝。`refit_count` 统计成功重拟合窗口数。

相关：[testing.md](testing.md)（回测窗口约定）、[preprocessing.md](preprocessing.md)（分解）。
