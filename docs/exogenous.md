# 外生与面板

## 多源输入

主线 artifact 保持单目标 `target_col -> yhat`，但支持三类输入：

| 配置 | 含义 |
| --- | --- |
| `endog_cols` | 历史内生协变量列（不含 `target_col`） |
| `exog_cols` | 历史外生变量列 |
| `future_exog_path` + `future_exog_time_col` + `future_exog_cols` | 独立未来外生文件 |

```bash
.venv/bin/python run.py \
  --data_path /abs/path/history.csv --time_col ds --target_col y \
  --model_name linear_var --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' \
  --endog_cols load --exog_cols temp \
  --future_exog_path /abs/path/future_exog.csv --future_exog_time_col ds --future_exog_cols temp \
  --do_train true --do_test true --do_forecast true --history_size 12 --predict_horizon 4
```

## 模型能力门禁

- AR/MA/ARMA/ARIMA/SARIMA/AutoARIMA 真实消费历史与未来外生变量
- 回归型 ARIMA 将历史输入中除目标外的列作回归变量；预测必须提供相同列集合与正确行数，按训练列顺序对齐
- 不可把仅历史可得的协变量当作未来已知变量
- 不支持的历史协变量或未来输入默认失败；兼容旧实验可显式 `--ignore_unsupported_inputs true`（记录在配置与实验路径中，不增加模型能力）
- 自动选型按历史/未来输入与 native 能力过滤候选；回测披露 `perfect_foresight`，不伪造历史天气预报
- CLI 阶段错误以非零退出，错误与已有产物索引保留在 run summary

## 面板多序列、多模型

```bash
env -u PYTHONPATH .venv/bin/python run.py --data_path /abs/path/panel.csv --series_id_col id --batch_models '{"naive":{},"auto_ets":{"season_length":7}}' --forecast_strategy native
```

- 输入 `id, ds, y` 长表；同序列同时间不允许重复；未来外生长表须携带相同 ID 集合
- 按"序列 × 模型"串行运行完整单序列主线，配置、预处理与模型状态独立；不是共享参数的全局模型
- 分拆输入存 `results/{数据名}/results_train/batch_inputs/{batch_id}/`；各序列产物按 `{数据名}-series-{安全ID}` 分组；批量 manifest 与汇总存 `results/{数据名}/results_forecast/batch/{batch_id}/`
- 默认任一任务失败整批非零退出；`--batch_allow_failed true` 才允许部分结果并标注 `survivor_bias`
- 面板入口要求已聚合数据与显式候选，不与 auto_select、聚合前级或监控回填混用；EDA 每序列只跑一次；未引入 Spark/Ray/Dask
