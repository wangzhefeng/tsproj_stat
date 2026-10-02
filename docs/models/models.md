# 模型体系

## 结构

- 统一接口：`fit(y, X_hist=None, X_future=None) / predict(horizon, X_future=None)`；`predict_one()` 默认桥接单步
- 实现位于 `models/model/` 家族模块；registry 与能力声明位于 `models/registry.py`（能力位在 ModelSpec 构造处内联写全）；工厂位于 `models/factory.py`
- models 与 forecasting 共享的契约归 `models/contracts/`：序列形状（inputs）、预测长度（validation）、区间列名协议（intervals）、外生输入（exogenous）
- 后端布局、fallback 与输入/区间桥接详见 [后端契约](backends.md)；ARCH/GARCH 共享 `_ARCHFamilyModel` 模板
- 新增模型必须接入 factory + registry metadata 标明稳定性分层，能力声明参与运行门禁，并在 `models/model/__init__.py` re-export（`tests/test_registry_consistency.py` 自动校验）
- `runtime_info()` 由模型层声明后端状态，区分备用模型与实际降级；归档与完整性回读见 [artifacts.md](../artifacts/artifacts.md)。

## 家族与模型（29 个 / 7 家族）

| 家族 | 模型 |
| --- | --- |
| 基线 | `naive / seasonal_naive / historic_average / croston` |
| ARIMA | `ar / ma / arma / arima / sarima / auto_arima`（`models/model/arima_family.py`） |
| 指数平滑 | `ets`（SES/DES/TES 统一入口，见 [preprocessing.md](../data_provider/preprocessing.md#ets-季节周期来源)） |
| Theta | `theta / dynamic_theta / auto_theta` |
| 多变量 | `var / bayesian_var / linear_var` |
| 波动率 | `arch / garch` |
| 扩展 | `prophet / tbats / neuralprophet`（`models/model/extended_models.py`，依赖缺失显式 fallback） |
| StatsForecast 候选 | `sf_auto_arima / auto_ces / random_walk_drift / seasonal_window_average / auto_ets` |

`sf_auto_arima` 是独立 StatsForecast 后端，不替换 `auto_arima` 的 pmdarima 语义；两者比较必须固定数据、窗口、策略与搜索范围。

## 稳定性分层

registry 能力位包括：`supports_future_exog`、`supports_prediction_intervals`、`supports_native_multistep`、`supports_update`、`supports_fitted_values`。能力用于运行门禁，不凭方法名推断支持；训练诊断的开关、后端取值、原始尺度产物及限制见 [拟合值诊断](fitted-values.md)。

| 分层 | 数量 | 说明 |
| --- | --- | --- |
| `stable` | 12 | 默认主线与常规 baseline，`auto_select` 默认候选 |
| `optional` | 11 | 依赖额外库（statsforecast/prophet/tbats）；新候选不自动进入默认选型 |
| `experimental` | 6 | `croston / neuralprophet / bayesian_tmt / bayesian_var / linear_var / rar`，需额外验证 |

- `models.stability.build_smoke_matrix()` 生成 optional/experimental 模型 smoke matrix（`success / dependency_unavailable / fit_failed`）
- `auto_select` 默认取 `stable` 候选；指标方向与偏差选优规则见 [testing.md](../evaluation/testing.md)
- `bayesian_tmt` 只表示"实验性单序列贝叶斯滞后回归近似"，与旧矩阵分解算法无关

## 运行脚本

wind 单变量与 AIDC 两路的按模型脚本见 [data.md](../data_provider/data.md#数据项目脚本)。`run_auto_arima.sh` 与 `run_sarima.sh` 定位为偏快日常脚本（缩短 history、增大 step、收紧搜索）。`run_tbats.sh`/`run_neuralprophet.sh` 的 smoke 可能经 fallback 完成，需看 `test_summary.json` 的 `used_fallback` 字段。

相关：[strategies.md](../forecasting/strategies.md)（策略与区间）、[exogenous.md](../pipeline/exogenous.md)（外生门禁）。
