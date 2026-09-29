# 已知限制

当前已知问题与技术债清单。历史问题台账与修复记录见 [LOG.md](LOG.md)；StatsForecast 扩展的边界与偏差记录见 [statsforecast-extension.md](statsforecast-extension.md)。

## 区间路径约束

- 当前区间路径要求 `scale=false`、`feature_mode=analysis_snapshot`；不把已有缩放或 `feature_mode=model_input` 路径当作已经过概率校准验证
- 面板入口不与自动选型、聚合前级和监控回填混用
- `interval_method=native` 在 `recursive / dirrec` 下已改为显式 RAISE（P6，2026-09-29；config.validate 与 predict_frame 双重拦截并提示 conformal 替代）
- 底层接口 `Forecaster.forecast_with_intervals()` 同样对 recursive/dirrec RAISE——config 层拦截只覆盖 CLI 主链路，直接调用该接口同样会得到显式错误
- conformal 在 recursive 下的校准半径随步长递增（逐步误差累积的顺序统计量），属预期行为而非缺陷

## 预处理已知耦合（T18 记录，暂不影响主线正确性）

- 分解模式的未来趋势外推为常数（`_future_trend` 平推 `_last_trend`），长 horizon 下趋势项不再增长；需要趋势外推时用 `detrend_method=linear`
- `detrend_method=moving_average` 的趋势窗口复用 `denoise_window`（默认 3），跨职责参数耦合；需要独立趋势窗口时当前无单独参数

## 告警治理

- ARIMA 家族仍可能出现少量 `ConvergenceWarning`；后续治理保持模型层定向处理，不回到入口层全局过滤
- 自动季节周期推断默认优先 ACF 峰值，再回退频域候选；季节性不明显时退回无季节路径

## 类型检查

- Pyright 对比 HEAD 基线后仅一项新增诊断（StatsForecast `predict(level)` 注解为 `List[int]` 与本项目小数置信水平的冲突），保留原有类型债；不得声明"类型检查通过"。详见 [statsforecast-extension.md](statsforecast-extension.md#验证记录)

## 环境一致性

- 若 `.venv` 与 `pyproject.toml` / `uv.lock` 不一致，先 `uv sync --extra dev` 再开发（见 [setup.md](setup.md)）
