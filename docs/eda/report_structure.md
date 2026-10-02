# EDA 报告结构与字段

## 3. 八段结构

1. **标题** + **技术摘要**（6–8 条结论）
2. **§1 数据口径与质量** — 频率、记录数、补齐点数、重复、缺失率 + 填充策略说明
3. **§2 趋势与水平** — 描述统计、trend_strength、residual_std、forecastability
4. **§3 季节性与周期** — seasonal_strength、CH/OCSB、ACF 峰值、FFT、谐波
5. **§4 平稳性与自相关** — ADF/PP/KPSS 表、ACF1/PACF1、Ljung-Box
6. **§5 波动与异常** — ARCH/White/BP、BDS、异常点
7. **§6 建模建议** — 来自 recommendations.json
8. **§7 限制** / **§8 后续验证** / 数据来源页脚

---

## 4. 数据字典（字段 → 消费段落）

| JSON 字段 | 段落 |
| --- | --- |
| `summary.{n_samples,mean,std,min,max,q1,q3,skewness,kurtosis}` | §2 |
| `data_quality.missing_ratio` | §1 |
| `summary.decomposition.{period,trend_strength,seasonal_strength,residual_std}` | §2/§3 |
| `summary.cycle.{dominant_period_fft,acf_peak_lags}` | §3 |
| `summary.seasonal_diff.{D_ch,D_ocsb}` | §3 |
| `summary.white_noise.{ljung_box_stat,ljung_box_pvalue}` | §4 |
| `summary.stationarity[]` | §4 |
| `summary.acf_head/pacf_head` | §4 |
| `summary.heteroskedasticity.{arch_lm_*,white_pvalue,bp_pvalue}` | §5 |
| `summary.stochasticity.{bds_*}` | §5 |
| `summary.outliers.*` | §5 |
| `summary.forecastability` | §2/§7 |
| `recommendations.{seasonal_period,differencing,preprocessing,model_family}` | §6 |
| `data_quality.*` / 审计 `{source_rows,output_rows,inserted_timestamp_count,duplicate_timestamp_count,time_range_*}` | §1 |

---

返回 [报告指南目录](eda_report_guide.md)。
