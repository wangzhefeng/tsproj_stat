# 预测引擎组件契约

本页记录当前组件职责；策略语义、区间算法与示例见 [策略与区间](strategies.md)。

| 当前组件 | 职责 |
| --- | --- |
| `forecaster.py` | Forecaster 接口组装 |
| `strategies.py` | native/single_step/direct/recursive/dirrec 多步推理 |
| `origins.py` | `prepare_origin_inputs` 修复与预处理、`forecast_at_origin` 原点编排、`rolling_error_pool` 带符号误差池 |
| `intervals.py` | IntervalSpec、组合裁决、conformal/native 区间 |
| `simulation.py` | 误差驱动样本路径模拟 |

## 区间与模拟

- `IntervalSpec` 包含 method/alpha/conformal_n_windows/levels；`resolve_interval_plan` 裁决区间方法与策略组合。
- `native × recursive/dirrec` 在 config.validate 与 predict_frame 双重 RAISE；直接 Forecaster 区间入口也检查，不输出 NaN 边界冒充区间。
- `IntervalSpec.levels` 为空时回退 `[1-interval_alpha]`；任一置信水平不可达即整体失败，不裁剪或丢弃水平。
- 列名及后端多水平桥接复用 [模型共享契约](../models/backends.md#输入与区间契约)。
- conformal 与 simulation 的误差池来源共用 rolling_error_pool；每窗独立预处理，误差保留原始尺度，不使用最终测试段或原点外数据。
- `simulate_enabled` 默认 false；配置为 `simulate_n_paths`、`simulate_n_windows`、`simulate_error_distribution`、`simulate_quantiles`，随机种子沿用 config.seed。
- 区间与模拟可同时开启，各自计算并验证产物；不把误差重采样称为模型后端原生 simulate。

## 固定参数前向更新

- `forecast_use_update=true` 仅用于 recursive；首步 fit，后续将预测追加历史并以固定参数 update 滤波。
- 门禁要求模型 supports_update、无未来外生，且不与派生模型输入组合；不满足即 RAISE。
- 默认 false，保持旧逐步重拟合语义；不把减少拟合次数视为与旧预测结果必然相同。
- 回测用真实历史重新滤波，与此处追加预测不同，见 [重拟合与状态更新](strategies.md#重拟合与状态更新)。
