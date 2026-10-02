# EDA 子系统

入口 `eda/pipeline.py`，产出结构化摘要、诊断表、图表路径与建模建议；数据项目级流程，不随模型运行。

新增诊断接入同一 pipeline；`eda/input_view.py` 适配输入，`eda/writers.py` 落结构化产物，`eda/visualization.py` 用 Agg 后端绘图。单检验失败返回 `ok=False` 与 error，不中断其他诊断；输入门禁仍直接失败。

输入必须为有限目标、排序唯一且匹配 freq 的时间轴；EDA 不再插值或补时间戳，缺失/不规则输入明确失败。需要离线修复时先通过数据层显式准备并保存审计，再传入分析视图。摘要 `input_view.policy=as_provided` 仅描述本次 EDA 未改动输入，不代表上游没有修复；全量 EDA 分量不得复用于回测训练。

主分析入口 `input_view.prepare_series` 复用 `data_provider/quality/checks.py` 的有限值与规则时间原语，错误保留 `EDA input` 上下文；不少于10个样本的 EDA 专属门禁仍归输入适配层。空输入由共用质量门禁拒绝，不补轴、不排序、不插值。

## 诊断能力

| 类别 | 方法 |
| --- | --- |
| 平稳性 | ADF / KPSS / PP |
| 分解 | STL（趋势/季节/残差强度）；配置周期与 ACF 峰值形成的候选周期 ≥2 时追加 MSTL 多周期强度 |
| 周期 | FFT 主周期 + ACF 峰值候选 |
| 季节差分建议 | CH / OCSB |
| 协变量 | 配置 `endog_cols`/`exog_cols` 时附带同期相关、CCF 最佳领先滞后、Granger（x→y）；建议层 Granger p<0.05 入 useful，单协变量失败结构化不中断 |
| 异方差 | ARCH-LM / White / Breusch-Pagan |
| 白噪声 | Ljung-Box |
| 非线性 | BDS |
| 离群点 | IQR / Z-score |
| 可预测性 | 谱熵归一化分数 |
| 建模建议 | `eda_recommendations.json/csv`：季节周期、差分、预处理、模型族候选、可用协变量 |

图集：序列/一阶差分/分布/周期图/STL 三分量/季节子序列（槽位分布剖面，样本 ≥2 个周期时产出）/ACF-PACF。

## 运行

```bash
.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false --eda_period 7 --eda_nlags 24 --eda_recommendation_enabled true
```

已接入数据项目用独立脚本（见 [data.md](../data_provider/data.md#数据项目脚本)）。多序列比较用 `--eda_comparison_paths` + `--eda_comparison_labels`。对预处理后序列再跑一轮 EDA 用 `--eda_run_preprocessed true`（见 [preprocessing.md](../data_provider/preprocessing.md)）。

## 自动报告

- 每次 `run_eda.sh` 自动生成中文叙述报告 `EDA_REPORT.md`（8 段，由 `eda/report_generator.py` 渲染，复用 recommendations 不重算阈值）
- 默认 `eda_generate_report=true`；手写报告（无 auto-generated marker）默认保留跳过，`--eda_report_overwrite true` 强制覆盖
- 生成原理、数据字典、降级矩阵与精修流程见 [eda_report_guide.md](eda_report_guide.md)

## EDA 输出路径

`results/{data_name}/results_eda/{eda_path}/`：按频率、周期、nlags、建议开关、预处理开关与聚合语义独立分组，与模型实验路径解耦（见 [usage.md](../config/usage.md#输出目录)）。
