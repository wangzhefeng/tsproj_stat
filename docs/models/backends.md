# 模型后端与共享契约

本页描述当前实现位置，不冻结文件布局；拆分或迁移须保持接口、唯一实现及归档兼容性。

## 家族实现

- `models/model/arima_family.py` 内聚 ARIMA 家族及阶数搜索；参数变体复用模型参数，不另建模型入口。
- `auto_arima` 保持 pmdarima 语义；`sf_auto_arima` 使用 StatsForecast，fit 失败回退 `ARIMAModel(auto_order=True)`，不静默替换二者身份。
- StatsForecast 的 `_StatsForecastModelBase`、`statsforecast_levels_frame`、`StatsForecastAutoARIMAModel` 当前归 `models/model/statsforecast_backend.py`；`baseline_models.py` 保留手写基线。
- 同一后端文件还包含 `auto_ets / auto_theta / dynamic_theta / auto_ces / random_walk_drift / seasonal_window_average`，不在各家族复制 SF 桥接。
- `ETSModel` 统一 SES / DES / TES，不平行新增 ses/des/tes 入口；statsmodels 后端支持 `damped_trend`，要求 trend 非空。季节周期与 smoothing grid 见 [预处理](../data_provider/preprocessing.md#ets-季节周期来源)。
- `prophet / tbats / neuralprophet` 当前归 `extended_models.py`；依赖缺失或运行时不兼容须显式 fallback 或可读报错。
- `bayesian_tmt` 只表示单序列贝叶斯滞后回归近似，不复活旧矩阵分解接口；模型清单、稳定性与 smoke 状态见 [模型体系](models.md)。
- 趋势/季节的预处理及重组统一走 TargetTransformer，不在模型家族里重复实现应用级变换链。

## 输入与区间契约

| 当前实现 | 职责 |
| --- | --- |
| `models/contracts/inputs.py` | 通用输入形状 |
| `models/contracts/validation.py` | 预测长度 |
| `models/contracts/exogenous.py` | 历史/未来协变量契约 |
| `models/contracts/intervals.py` | `interval_bound_columns` / `iter_bound_pairs` / `resolve_interval_levels` |

- models 不 import data_provider；共享周期推断归 `utils/seasonality.py`，有限值门禁不反向引入数据层。
- 区间列名协议由 forecasting re-export 保留旧导入路径，不复制实现。
- SF 多水平区间重写 `predict_with_levels`，一次调用后端；其余模型由 `BaseStatModel` 逐水平委托 `predict_with_intervals`，不重复拟合。
- registry 在 `ModelSpec` 构造处声明能力，由 `tests/test_registry_consistency.py` 校验导出与能力一致性；方法存在不等于已声明能力。
- 持久化不归 models，统一由 [产物层](../artifacts/artifacts.md#checkpoint) 提供；移动可 pickle 类时必须验证旧归档回读。
