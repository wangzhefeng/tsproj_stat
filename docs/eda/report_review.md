# EDA 报告判读与精修

## 5. 叙述解释决策表（生成器内编码，p 值统一 α=0.05）

> 这些规则**不在** recommendations.json，是生成器/精修者的判读依据。

| 主题 | 规则 |
| --- | --- |
| 平稳性方向 | ADF/PP 原假设=单位根：`p>0.05`→不能拒绝(非平稳)；KPSS 原假设=平稳（**反向**）：`p<0.05`→拒绝平稳(非平稳)。`ok=False`/NaN→检验失败 |
| ARCH vs White/BP | arch<0.05 且 white/bp≥0.05 → 波动聚集（条件异方差），非固定异方差；arch<0.05 且 white/bp<0.05 → 部分可由时间趋势解释；arch≥0.05 → 未检测到 |
| FFT 伪周期 | `abs(fft−n)≤1` 或 `fft/n>0.9` → 主周期≈样本长度，是趋势/边界伪周期，非业务周期 |
| 谐波 | `acf_peak % 配置周期==0 且 >周期` → 更可能是周季节性的倍频/谐波（如 35=5×7） |
| Ljung-Box | `p<0.05` → 拒绝白噪声，含可利用依赖，残差须复检 |
| BDS | 任一 dim `p<0.05` → 不符合 iid，可能含线性外结构；但 BDS 不能证明具体非线性机制，须对残差复检 |
| forecastability | 谱熵相对评分（[0,1]），仅作复杂度辅助信号，**非可达预测准确率** |
| 异常点处置 | `max(rate)<0.03` → 异常率低，不默认全局去噪，先定位日期核对；否则核对后考虑去噪 |
| 置信度 | high/medium/low → 高/中/低 |

**复用项（已在 `eda/recommendations.py` 应用阈值，不要重算）**：`recommended_period`（seasonal_strength≥0.25 取配置周期，否则 ACF 峰值，否则 FFT≥2）、`recommended_d`（ADF/PP/KPSS 多数票）、`recommended_D`（max(D_ch,D_ocsb)）、`preprocessing`（trend≥0.35 线性去趋势；seasonal≥0.35 季节分解；outlier_rate≥0.03 moving_median）、`model_family`（theta 需 forecastability≥0.2）。

---

## 6. 风格与语气规则

- **克制**：不下绝对结论；低置信项要标注；周期/模型选择一律建议**样本外/rolling 回测**验证。
- **不主张因果**：ARCH/BDS/相关性是统计结构，不是因果。
- **引用图**：用相对路径 `plots/*.png`，仅引用确实存在的图。
- **forecastability ≠ 准确率**；ARCH/BDS 在原始水平序列上的显著性必须在**残差**上复检。
- 领域名词（如“负荷”）由人/Agent 补，生成器只用 `data_name`/`target_col`/`aggregation_method` 字面值。

---

## 7. 陷阱

- 标题与“日值定义”为字面值（如 `# A_Loads_1day EDA 报告`、`当日最大值`），不含领域名词——需精修。
- `data_quality.json` 仅在 **EDA-only** 运行时落 in eda_dir；模型运行时在 `train_results_dir`，§1 质量行会降级。
- 生成时 `run_summary.json` 尚不存在（它在 EDA 之后才写），不能作为生成输入。
- FFT/谐波 caveat 见 §5；ARCH 在水平序列显著 ≠ 在残差显著。

---

## 8. Agent 如何用本指南精修草稿

1. 读已生成的 `EDA_REPORT.md` + 同目录 `eda_summary.json` / `eda_recommendations.json` / `eda_diagnostics.csv` + 审计 JSON。
2. **逐个数字核对** JSON（绝不编造）；生成器已保证一致，精修时若改动措辞不要改数字。
3. 修命名：标题、日值定义、领域名词（如把 `当日最大值` 改为 `当日最大负荷`）。
4. 收紧叙述：合并重复句、补充业务语境、把自动建议落实为具体实验计划。
5. 遵守 §6 风格规则；保留或改写 §7/§8 以贴合该数据集。
6. 若希望后续自动生成不再覆盖精修版：删除文件首行的 `auto-generated` marker 注释（生成器会把手写版识别为“已存在且无 marker”而跳过）。

返回 [报告指南目录](eda_report_guide.md)。
