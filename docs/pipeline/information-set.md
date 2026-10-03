# 信息集与防数据泄露

## 原点边界

- 一次预测只能使用截至原点可获得的观测和预报；历史窗口先切分，再修复、变换与构造特征。
- 普通历史表默认由调用方保证观测在时间戳时可用；项目不推断外部文件是否曾经被离线补缺或事后修订。
- `target_col` 禁止作为历史外生/未来外生；递归历史只追加预测目标，未来协变量不得覆盖目标。

## 聚合与评估标签

- 建模采用 `aggregation_fill_method=preserve`：固定频率完整桶保留观测统计量，不完整桶（含首尾边界）保留 NaN；不跨桶插值，不丢时间行。源时间视为左端网格点，输出以桶结束时刻标记可用时间（右标签）。
- 历史 NaN 在各原点窗口内修复；测试真值不修复。`backtest_missing_target_policy=exclude` 显式只评价观测点，记录排除数量；默认 raise，全缺评估窗口仍失败。
- 缺失与负荷工况相关时，观测子集的指标仍可能有选择偏差；不能把 exclude 成绩解释为未观测日期也已验证。
- `linear/seasonal_slot` 聚合只用于离线准备/EDA；建模前拒绝。已存在 sidecar 必须通过来源、输出、审计摘要校验且无未来填充。
- `require_aggregation_audit=true` 进一步拒绝无审计来源。AIDC 模型脚本开启它并写新 `*_observed_*` 名称；旧离线数据和旧结果不覆盖、不迁移。

## 未来外生

- `exog_future_known` 默认 false（与旧默认 true 不兼容）；真正提前已知的日历/计划量须显式声明 true，不可用该声明伪装未来天气实测。
- 天气等预报档案用 `future_exog_time_col` 表示有效时刻、`future_exog_issue_time_col` 表示发布时间，同一有效时刻允许多个版本。
- `FutureExogSource.at` 选择发布时间 <= 原点的最新版本；发布时间恰等原点允许，之后版本不可用。缺覆盖/非有限值/同版重复明确失败，不回退实测。
- 历史回测、自动选型、conformal 和模拟校准使用同一版本选择；最终预测也按实际原点筛选，不能只给文件中最新一版。
- 普通已知表仍须按未来时间精确对齐。预报表与历史表的时区须兼容；观测上报延迟、历史修订数据的版本治理仍由上游提供。

## 选型与内部调参

- 单/多模型 auto_select 先在历史前段选型并冻结选择，再在独立尾段滚动测试；尾段可以逐窗更新已揭晓的历史，不能参与改选。
- 尾段长度、评分标记见 [回测与选型](../evaluation/testing.md)。独立测试不等于重复查看尾段调参后仍保持独立。
- ETS 平滑网格经 `forecasting/tuning.py` 从原始历史切内部训练/验证；每候选独立修复和拟合变换，逆变换后与原始验证真值评分。
- 候选只把选定超参带到最终全窗拟合，不复用内部训练状态。模型归档记录 selected_smoothing_params/tuning_metadata；缺失验证真值明确失败。

## 验证

- 固定历史，只扰动原点后原始数据或之后发布的预报，该原点预测必须不变；聚合前级也必须参与实验。
- 内部验证尾段只允许影响评分/选参，不允许影响候选训练变换；独立外层测试不得影响选型。
- `tests/test_observed_aggregation.py`、`test_future_availability.py`、`test_selection_holdout.py`、`test_ets_tuning_boundaries.py` 与 `test_leakage_guards.py` 覆盖这些边界；执行规范见 [tests](../tests/README.md)。
