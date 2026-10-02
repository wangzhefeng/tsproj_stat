# EDA_REPORT.md 生成与精修指南

本文件是 `eda/report_generator.py` 的参考入口：先了解报告用途，再按章节查生成机制、字段和精修规则。

## 目的与范围

- `run_eda.sh` 跑完后，结构化产物为 `eda_summary.json`、`eda_diagnostics.csv`、`eda_recommendations.json` 和图表；`EDA_REPORT.md` 将其组织为可读报告。
- 报告保留原八段分析，§9 为任务门禁与配置建议，§10 为诊断证据，§11 为来源/月度/滚动/异常事件；不依赖历史报告。
- 自动报告是草稿：数字与统计判读来自生成器，维护者仍需逐项核对来源，并补充命名、措辞及领域语境。

## 章节目录

| 章节 | 内容 |
| --- | --- |
| [生成机制](report_generation.md) | 输入产物、调用方式、覆盖策略、缺失输入的降级矩阵 |
| [结构与数据字典](report_structure.md) | 报告章节、JSON/CSV 字段对应段落 |
| [证据与计算契约](evidence.md) | 共享计算、周期验证、异常明细、BDS 与模型边界 |
| [判读与精修](report_review.md) | 统计解释决策表、语气规则、常见陷阱及人工/Agent 精修步骤 |

生成器沿用此入口路径；具体内容在小章节中维护，不在本页重复。

返回 [EDA 模块](eda.md) · [项目主目录](../../README.md)。
