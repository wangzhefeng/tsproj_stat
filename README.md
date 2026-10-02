# 统计模型时间序列预测框架

统一支持统计模型训练、滚动回测、预测与 EDA；以 `run.py` 为唯一建模 CLI 入口。

本页是项目主目录：概要在这里，具体说明统一放在 `docs/`，通过链接逐层阅读。

## 开始使用

[安装与运行环境](docs/utils/setup.md) → [配置与运行命令](docs/config/usage.md) → [测试与验证](docs/tests/README.md)。

## 模块文档

| 模块 | 章节入口 | 内容 |
| --- | --- | --- |
| `config` | [运行与 CLI](docs/config/usage.md) | 配置来源、入口参数、阶段开关 |
| `pipeline` | [编排](docs/pipeline/README.md)、[外生与面板](docs/pipeline/exogenous.md)、[运行限制](docs/pipeline/limitations.md) | 窗口、阶段调度、多模型与批量 |
| `data_provider` | [数据与聚合](docs/data_provider/data.md)、[职责与接口](docs/data_provider/data-architecture.md)、[目标变换](docs/data_provider/preprocessing.md) | 接入、清理、质量、聚合、可逆变换 |
| `features` | [派生特征](docs/features/features.md) | 分析快照、窗口内派生、未来推进 |
| `models` | [模型体系](docs/models/models.md) | 模型家族、工厂、能力与稳定性 |
| `forecasting` | [策略与概率预测](docs/forecasting/strategies.md) | 多步策略、区间、校准、模拟 |
| `evaluation` | [回测与指标](docs/evaluation/testing.md) | 窗口评估、评分、模型选优 |
| `artifacts` | [产物协议](docs/artifacts/artifacts.md) | 身份、目录、序列化、归档、manifest |
| `monitoring` | [监控闭环](docs/monitoring/monitoring.md) | 预测记录、实际回填、滚动指标 |
| `eda` | [EDA](docs/eda/eda.md)、[报告指南](docs/eda/eda_report_guide.md) | 诊断、图表、建议及报告章节目录 |
| `utils` | [公共工具](docs/utils/README.md)、[运行环境](docs/utils/setup.md) | 日志、种子、缓存、演示数据、周期推断 |
| `tests` | [验证约定](docs/tests/README.md) | 测试命令、覆盖边界与副作用 |

## 协作

功能修改必须同步相关文档；维护者还需定期核查文档与实现是否一致，规则见 [AGENTS.md §4](AGENTS.md#4-文档同步与检查)。
