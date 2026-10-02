# 训练拟合值与残差诊断

## 开关与产物

- `train_fitted_values` 默认 false；显式开启要求 registry `supports_fitted_values`，未声明即 RAISE。
- train 阶段返回诊断数据，pipeline 负责写入本次 `results_train` 下的 `fitted_values.csv`，列为原始尺度 `y/fitted/residual`。
- `train_summary.residual_stats` 记录 mean/std/n 与 Ljung-Box p 值；标准差使用样本标准差。
- Ljung-Box 默认 lag 为 `min(10,n//5)`；样本少于 8 时 p 值为 None，不伪造检验结论。

## 后端与尺度

- statsmodels 系取 `fittedvalues`；StatsForecast 系需重传训练序列，经 `forecast(fitted=True)` 取得拟合值。
- theta 的 statsmodels 后端无此值，不声明能力；naive 基线当前也未纳入，不能仅凭预测接口推断支持。
- 启用预处理时，用 `inverse_transform` 按训练索引还原，不用面向未来外推的 `inverse_forecast`。
- 原始历史目标必须有限，不能以修复后的观测冒充真实残差；原始序列与拟合值长度须对齐。
- `feature_mode=model_input` 丢弃 warmup 后若与原始历史错位，明确拒绝诊断。

## 运行与扩展

```bash
.venv/bin/python run.py --model_name arima --train_fitted_values true --do_train true
```

新增后端诊断须同时验证模型能力、原始尺度数值及产物；不能只断言文件存在。当前未支持的组合不是永久禁令，但补齐实现与数值测试前不得放开门禁。

返回 [模型体系](models.md)；变换流程见 [预处理](../data_provider/preprocessing.md)。
