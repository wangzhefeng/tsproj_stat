# 派生特征

## 两种模式

- `feature_mode=analysis_snapshot`（默认）：只导出 `analysis_feature_snapshot.csv`；时间特征、lag 和 `target_t_plus_*` 用于分析，不进入模型。
- `feature_mode=model_input`：时间特征与目标 lag 进入模型，绝不生成未来标签；模型须支持所需历史/未来协变量。不支持时默认失败，显式忽略不能当作特征已参与建模。

## 窗口与尺度契约

1. 先切各自历史窗口，修复历史缺失，再拟合目标变换器。
2. 在变换后的目标上派生 lag；协变量保持原尺度；时间特征为 `hour/dayofweek/month/dayofyear`。
3. `max(lags)` 个 warmup 行同步丢弃，不借窗口外数据、不反向补 lag；空训练窗、派生列与原始输入重名均失败。
4. 训练、回测、自动选型、模拟校准共用 `features/model_inputs.py` 的 `ModelFeatureSpec`；修复/变换仍分别归 cleaning/target_transforms。
5. 每个预测步的时间特征来自调用方提供的未来时间轴，lag 来自该窗口历史或已经生成的预测，不读取评估真值，也不冻结最后一行。

## 推理代价与边界

- `native` 仍只 fit 一次；为了构造未知的目标 lag，逐增未来输入前缀并调用 predict，不能假定 predict 总成本不随前缀增长。
- `direct` 保持按步重拟合的兼容语义；`recursive/dirrec` 将对应未来特征和预测一起追加到历史。
- 模拟校准也使用相同特征规则；每窗长度须足以覆盖 warmup 和模型自身滞后要求。
- `forecast_use_update`、`backtest_refit_every != 1` 不与派生输入组合；区间仍要求 `analysis_snapshot`（见 [limitations.md](../pipeline/limitations.md)）。
- checkpoint 保存已拟合模型与目标变换器，不保存应用编排器。独立回读必须显式提供训练时同名、同尺度的未来协变量；动态 lag 需逐步构造。模型预测后再按 transformer.enabled 还原原始尺度；主线推理不读取 checkpoint。

## 示例

```bash
.venv/bin/python run.py --model_name linear_var \
  --model_params '{"target_lags":[1],"feature_lags":[0]}' \
  --forecast_strategy native --feature_mode model_input \
  --enable_datetime_features false --lags 1,2 --history_size 40 --predict_horizon 3
```

该例验证派生 lag 通路，不代表效果优于原模型。历史/未来外生约定见 [外生与面板](../pipeline/exogenous.md)，数值与实际输入验收规则见 [tests](../tests/README.md)。
