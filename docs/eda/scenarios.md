# EDA 数据场景与任务边界

## 场景配置

- 主入口 `run_eda.py` → 通用运行器 `eda/runner.py` → 场景 YAML → 既有 EDA-only 分析/报告/manifest；不调用 run.py。
- 配置位于 `scripts/aidc_power_month/route_A|route_B/eda/{15min,h,D}.yaml`、`scripts/wind/eda/D.yaml`、`scripts/ett_small/ETTm1/eda/15min.yaml`；同目录 run_eda.sh 为入口，复用 AppConfig。
- 相对配置/数据/结果路径始终以项目根为基准；shell 仅选择默认配置并透传 CLI，新入口要求显式 --config。
- 运行 `.venv/bin/python run_eda.py --config scripts/aidc_power_month/route_A/eda/15min.yaml`；通用运行器复用 ModelApp 的 EDA-only 分支，不执行模型阶段。
- 专用入口拒绝训练/回测/预测、自动选模、面板、监控、模拟和聚合；原 run.py 的兼容接口不在本次删除范围。
- AIDC EDA 读取既有 derived CSV，不执行聚合。缺失输入直接失败；数据准备仍由 `prepare_data.py` 独立执行。
- 常规场景显式 `eda_bds_mode: off`；深入诊断手动覆盖 full/tail，通用 AppConfig 默认 full 不变。

## 预测任务

- `eda_task_confirmed=false`：业务任务未确认，报告只输出参数约束，不继承默认 history/horizon 生成完整 YAML。
- 明确设置 history_size/predict_horizon 后，可用 `eda_task_confirmed=true` 声明其为业务选择；报告不推断最优值。
- 所有模型候选共用相同历史与评估窗口；样本不足以支持周期时省略该候选，不单独扩大它的训练窗口。
- 报告仍不执行选型、回测或自动回写模型配置。

## 分段与异常

- `eda_window_size/eda_window_step` 为点数，均0关闭滚动窗口；开启时 window≥10，1≤step≤window。
- 月度统计按时间戳自然月分组；滚动只取完整窗口并覆盖最新完整窗口，不把尾部短窗口当完整。
- 周期剖面验证的形状相关与双向R²分别保存，固定剖面未通过不等于不存在周期。
- `eda_acf_nlags` 可独立扩大 ACF 范围；PACF 仍按 eda_nlags，输出超PACF范围的列为空，不伪造零值。
- `eda_local_outlier_window` 为奇数点，0关闭；滚动中位数和MAD使用居中窗口，边缘至少半窗。这是离线EDA，不是as-of检测。
- 异常事件按同一方法的相邻标记合并，跨度按含末采样间隔计；不自动删点、替换或诊断设备故障。

## 来源证据

- 自动发现 CSV 旁的 `文件名.csv.aggregate.json`，只读校验，不补审计、不重建CSV。
- 核对v2版本、审计摘要、CSV摘要、源文件摘要（可访问时）、目标频率与列名；状态和审计文件指纹进入本次结果。
- 缺失、损坏、来源不可访问不伪装 verified；缺失审计仍可做数值EDA，但上游填补/as-of状态未知。
- 审计哈希仅证明内容一致性，不提供来源认证；双向补值标记不因当前CSV无缺失而消失。

返回 [EDA](eda.md) · [配置](../config/usage.md)。
