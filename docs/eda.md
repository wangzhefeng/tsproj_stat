# EDA 子系统

入口 `eda/pipeline.py`，产出结构化摘要、诊断表、图表路径与建模建议；数据项目级流程，不随模型运行。

输入必须为有限目标、排序唯一且匹配 freq 的时间轴；EDA 不再插值或补时间戳，缺失/不规则输入明确失败。需要离线修复时先通过数据层显式准备并保存审计，再传入分析视图。摘要 `input_view.policy=as_provided` 仅描述本次 EDA 未改动输入，不代表上游没有修复；全量 EDA 分量不得复用于回测训练。

## 诊断能力

| 类别 | 方法 |
| --- | --- |
| 平稳性 | ADF / KPSS / PP |
| 分解 | STL（趋势/季节/残差强度） |
| 周期 | FFT 主周期 + ACF 峰值候选 |
| 季节差分建议 | CH / OCSB |
| 异方差 | ARCH-LM |
| 白噪声 | Ljung-Box |
| 可预测性 | 谱熵归一化分数 |
| 建模建议 | `eda_recommendations.json/csv`：季节周期、差分、预处理、模型族候选 |

## 运行

```bash
.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false --eda_period 7 --eda_nlags 24 --eda_recommendation_enabled true
```

已接入数据项目用独立脚本（见 [data.md](data.md#数据项目脚本)）。多序列比较用 `--eda_comparison_paths` + `--eda_comparison_labels`。对预处理后序列再跑一轮 EDA 用 `--eda_run_preprocessed true`（见 [preprocessing.md](preprocessing.md)）。

## 自动报告

- 每次 `run_eda.sh` 自动生成中文叙述报告 `EDA_REPORT.md`（8 段，由 `eda/report_generator.py` 渲染，复用 recommendations 不重算阈值）
- 默认 `eda_generate_report=true`；手写报告（无 auto-generated marker）默认保留跳过，`--eda_report_overwrite true` 强制覆盖
- 生成原理、数据字典、降级矩阵与精修流程见 [eda_report_guide.md](eda_report_guide.md)

## EDA 输出路径

`results/{data_name}/results_eda/{eda_path}/`：按频率、周期、nlags、建议开关、预处理开关与聚合语义独立分组，与模型实验路径解耦（见 [usage.md](usage.md#输出目录)）。
