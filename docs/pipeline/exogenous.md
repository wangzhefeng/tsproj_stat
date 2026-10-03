# 外生与面板

## 多源输入

当前主线 artifact 为单目标 `target_col -> yhat`，支持三类输入；多目标扩展须先定义接口与产物契约，不视为现成能力：

| 配置 | 含义 |
| --- | --- |
| `endog_cols` | 历史内生协变量列（不含 `target_col`） |
| `exog_cols` | 历史外生变量列 |
| `future_exog_path` + `future_exog_time_col` + `future_exog_cols` | 独立未来外生文件 |

Loader 读取全部未来输入；确定性已知表按预测时间选择，预报档案由 `data_provider/availability.py` 按 valid time + issue time 选择原点前最新版本。覆盖不足、选中值缺失、同版重复均失败。

```bash
.venv/bin/python run.py \
  --data_path /abs/path/history.csv --time_col ds --target_col y \
  --model_name linear_var --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' \
  --endog_cols load --exog_cols temp \
  --future_exog_path /abs/path/future_exog.csv --future_exog_time_col ds --future_exog_cols temp \
  --future_exog_issue_time_col issued_at \
  --do_train true --do_test true --do_forecast true --history_size 12 --predict_horizon 4
```

## 模型能力门禁

- AR/MA/ARMA/ARIMA/SARIMA/AutoARIMA 真实消费历史与未来外生变量
- 回归型 ARIMA 将历史输入中除目标外的列作回归变量；预测必须提供相同列集合与正确行数，按训练列顺序对齐
- `exog_future_known` 默认改为 false；日历/计划量须显式 true，天气等须提供 `future_exog_issue_time_col`。历史回测需要覆盖每个原点的预报档案，不接受未来实测冒充预报。
- 目标列不得出现在 `exog_cols/future_exog_cols`；配置与公共推理双重拒绝，递归追加也禁止同名覆盖。
- 不支持的历史协变量或未来输入默认失败；兼容旧实验可显式 `--ignore_unsupported_inputs true`（记录在配置与实验路径中，不增加模型能力）
- 回测、选型、conformal、模拟共同选择发布时间 <= 原点的版本；策略披露 known_in_advance/as_of_forecast/none，详见 [信息集](information-set.md)。
- CLI 阶段错误以非零退出，错误与已有产物索引保留在 run summary

## 面板多序列、多模型

```bash
env -u PYTHONPATH .venv/bin/python run.py --data_path /abs/path/panel.csv --series_id_col id --batch_models '{"naive":{},"auto_ets":{"season_length":7}}' --forecast_strategy native
# 任务级续跑（需已有 batch manifest）：追加 --batch_resume_from /abs/path/to/上次/batch_manifest.json
```

- 输入 `id, ds, y` 长表；同序列同时间不允许重复；未来外生长表须携带相同 ID 集合
- 按“序列 × 模型”隔离执行单序列主线；`batch_n_jobs=1` 串行，>1 进程并行。历史/未来帧内存直通，审计 CSV 不读回；不是共享参数的全局模型
- 续跑要求完整配置（除续跑路径/容忍开关）与历史/未来输入内容指纹一致；成功子任务须通过 manifest 文件校验、序列身份和必需输出校验。缺指纹的旧 manifest 拒绝续跑；失败任务正常重试，子任务不继承父级续跑控制
- 进程被强制中断且尚无 batch manifest 时不能直接恢复；不是模型级 checkpoint 续训。内存未来帧校验显式使用 `future_exog_available`，不伪造文件路径
- 分拆输入存 `results/{数据名}/results_train/batch_inputs/{batch_id}/`；各序列产物按 `{数据名}-series-{安全ID}` 分组；批量 manifest 与汇总存 `results/{数据名}/results_forecast/batch/{batch_id}/`
- 默认任一任务失败整批非零退出；`--batch_allow_failed true` 才允许部分结果并标注 `survivor_bias`
- 面板入口要求已聚合数据与显式候选，不与 auto_select、聚合前级或监控回填混用；EDA 每序列只跑一次；未引入 Spark/Ray/Dask
