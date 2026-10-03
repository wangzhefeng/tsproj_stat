# evaluation：回测、指标与选型

## 模块职责

| 文件 | 职责 |
| --- | --- |
| `backtest.py` | 窗口切分、逐窗推理、明细/分步/跨窗口指标汇总 |
| `metrics.py` | 点指标注册表、缩放基准与区间评分 |
| `comparison.py` | 多模型比较表组装、排名和选优的纯计算 |
| `selector.py` | 按候选模型运行有窗口数上限的回测并选优 |

## 滚动回测

- `do_test` 承担历史评估；窗口模式 `expanding / sliding`，训练长度由 `backtest_train_size` 控制。
- 未显式设置时训练长度等于 `history_size`；sliding 保持长度，expanding 仅初始窗等长、后续扩张。解析优先级为 `backtest_train_size` > 旧字段 `backtest_initial_train_size` > `history_size`。
- `backtest_n_jobs>1` 按窗口并行，输出按 `window_id` 稳定排序；固定参数更新限制见 [预测策略](../forecasting/strategies.md#重拟合与状态更新)。
- 每窗修复/预处理/特征构造只看历史；评估与校准真值不插值；建模拒绝已知双向补缺来源。
- `backtest_missing_target_policy=raise` 默认拒缺失标签；显式 `exclude` 只对观测点评分，预测明细仍保留 NaN 真值，披露 observed/excluded_target_count；全缺窗口仍失败。
- 窗口失败默认 RAISE；显式 `backtest_allow_failed_windows=true` 才容忍，汇总标记 `survivor_bias`，失败窗口另存。
- 输出窗口级预测/指标、`backtest_step_metrics.csv`、汇总及预测对比/残差/误差分布图；落盘由 pipeline 收口。

## 指标与选择

- 点指标唯一事实来源为 `evaluation/metrics.py` 的 `POINT_METRICS`（func/higher_is_better/requires_train）；回测列、分步与跨窗汇总、选型白名单与比较表排序均由它派生。
- `mase/rmsse` 以窗口训练序列 naive m 阶差分为缩放基准；常数窗等基准不可用记 NaN，汇总跳过。
- `r2` 越大越好；`bias` 按绝对值越小越好，报告保留正负号；其余误差越小越好。
- JSON null/NaN/Inf 不参与选优，对比表置尾；全无有效评分时失败。
- 区间输出 coverage/width/Winkler；多水平按水平后缀展开。Winkler 使用实际有效置信水平，见 [区间约定](../forecasting/strategies.md)。
- `auto_select` 使用原始历史前段选型，独立尾段只用于最终滚动评估；单/多模型同规则，选型失败关闭后续运行，候选不得靠跳过失败窗口取胜。
- `auto_select_holdout_size` 指定尾段点数，默认 max(horizon, ceil(20% × 总行数))；前段须足够完成选型。摘要记录 selection_end/holdout_start；测试标记 independent_holdout。尾段滚动时允许纳入此前已揭晓的标签，但不改选模型。
- 不开启选型时仍是原来的完整滚动评估；多模型对比的尾段排名不能再用来改选。配置用法见 [config](../config/usage.md)。

自动化测试、命令与覆盖约定独立归 [tests](../tests/README.md)，本页不维护单次验收数字。
