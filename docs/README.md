# tsproj_stat 文档索引

所有项目文档统一收纳于 `docs/`，按主题分小文件维护，遵循渐进式披露：每个文件只讲一个主题，控制在 60 行以内，详细内容链接到对应小文件。

根目录另有 `AGENTS.md`（AI 编码工具的项目协作规范，必须留在仓库根目录）与 `README.md`（面向人的快速索引）。

## 快速导航

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| 问题台账 | [LOG.md](LOG.md) | 项目状态、当前问题、修复记录、验证记录 |
| 安装与依赖 | [setup.md](setup.md) | uv 虚拟环境、依赖管理、缓存纪律 |
| 运行与 CLI | [usage.md](usage.md) | 运行示例、CLI 参数、阶段开关 |
| 数据与聚合 | [data.md](data.md) | 数据集布局、频率聚合、派生数据与审计 |
| 预处理 | [preprocessing.md](preprocessing.md) | 去噪、去趋势、可逆分解、逆变换 |
| 模型体系 | [models.md](models.md) | 家族结构、模型清单、稳定性分层 |
| 策略与区间 | [strategies.md](strategies.md) | 预测策略、预测区间、校准 |
| 外生与面板 | [exogenous.md](exogenous.md) | 多源输入、外生变量、面板批量 |
| 回测与验证 | [testing.md](testing.md) | 滚动回测、指标、验证命令基线 |
| EDA 子系统 | [eda.md](eda.md) | 诊断能力、建议输出、报告生成 |
| 监控闭环 | [monitoring.md](monitoring.md) | 预测日志、实际值回填、滚动指标 |
| 已知限制 | [limitations.md](limitations.md) | 已知问题、技术债、清理时机 |
| EDA 报告精修 | [eda_report_guide.md](eda_report_guide.md) | EDA_REPORT.md 生成原理与精修指南 |
| 实施记录 | [statsforecast-extension.md](statsforecast-extension.md) | StatsForecast 扩展实施与验收记录 |

## 维护规则

- 修改主线功能（`app / config / models / evaluation / data_provider / features / eda / utils`）时，同步更新上表中受影响的主题文档；不确定改哪篇时更新 [LOG.md](LOG.md)。
- 新增文档统一放 `docs/`，先在本索引登记再写内容；单文件原则 ≤60 行，超了就按主题再拆。
- 文档间用相对链接互引，不复制内容；同一事实只在一处维护，其余地方链接过去。
