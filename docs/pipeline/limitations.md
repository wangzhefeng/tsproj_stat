# 跨模块运行限制

本页集中说明主流程可组合性与部署边界；具体模型能力见 [models](../models/models.md)，不以历史实施记录代替当前运行契约。

以下是当前实现限制，不是永久禁止扩展。新增组合或 checkpoint 推理模式必须明确契约、补齐实现与数值/集成测试并同步文档；此前保持门禁，不因改写规范就宣称支持。

## 区间路径约束

- 当前区间路径要求 `scale=false`、`feature_mode=analysis_snapshot`；不把已有缩放或 `feature_mode=model_input` 路径当作已经过概率校准验证
- 面板入口不与自动选型、聚合前级和监控回填混用
- `interval_method=native` 在 `recursive / dirrec` 下已改为显式 RAISE（P6，2026-09-29；config.validate 与 predict_frame 双重拦截并提示 conformal 替代）
- 底层接口 `Forecaster.forecast_with_intervals()` 同样对 recursive/dirrec RAISE——config 层拦截只覆盖 CLI 主链路，直接调用该接口同样会得到显式错误
- conformal 半径按各步长的校准误差独立取顺序统计量；可能随步长增大，但实现不保证单调递增。

## 当前预处理限制

- 分解模式的未来趋势外推为常数（`_future_trend` 平推 `_last_trend`），长 horizon 下趋势项不再增长；需要趋势外推时用 `detrend_method=linear`
- `detrend_method=moving_average` 的趋势窗口复用 `denoise_window`（默认 3），跨职责参数耦合；需要独立趋势窗口时当前无单独参数

## 告警治理

- ARIMA 家族仍可能出现少量 `ConvergenceWarning`；后续治理保持模型层定向处理，不回到入口层全局过滤
- 自动季节周期推断默认优先 ACF 峰值，再回退频域候选；季节性不明显时退回无季节路径

## 类型检查

- 按根 AGENTS.md §3 分级验证；类型检查范围与 CLI 清单见 [tests](../tests/README.md)，历史通过记录不代表当前代码已经验证。

## 本地产物边界

- v2 新路径不兼容硬编码旧文件位置；存量结果不迁移，历史运行持续占用磁盘。
- 单文件原子替换不等于整包事务或断电持久性保证；单次运行不支持阶段内恢复；面板支持已有 batch manifest 的任务级复用（见 [exogenous.md](exogenous.md)）。身份发布锁依赖本地 POSIX 文件系统。
- pickle 仅用于可信本地归档；checksum 不提供来源认证。累计监控日志不纳入不可变文件清单。
- 派生 lag 的 native 推理只 fit 一次，但逐增前缀 predict 会增加成本；不支持 fixed-parameter update，与区间的组合仍拒绝（见 [features.md](../features/features.md)）。监控回填仍须单写者串行。

## 环境一致性

- 本次开发依赖的 `.venv` 不一致时先 `uv sync --extra dev`；无关可选依赖或业务数据缺失不阻塞独立开发，但必须披露受限验证范围，不能宣称相关功能已验收（见 [setup.md](../utils/setup.md)）。
