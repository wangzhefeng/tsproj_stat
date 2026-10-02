# EDA 证据与计算契约

## 共享计算与产物

- `diagnostics.analyze_series` 返回 `DiagnosticResult`（内存对象）；原有 `run_diagnostics` 保留 summary/diagnostics 二元接口。
- `evidence.py` 负责时间单位、分析视图、周期分段筛选与异常明细；不改输入、不写文件。
- 原始、线性去趋势、一阶差分视图分别计算平稳性、ACF/PACF 和频谱；同一视图计算一次。
- STL 只拟合一次，摘要、CSV 和图使用同一分量；MSTL 是独立的多周期诊断。
- `eda_components.csv`：time/value/trend/seasonal/residual；分解失败时保留表头，不伪造分量。
- `eda_correlations.csv`：view/lag/acf/pacf 及各自置信区间；保存请求范围内完整相关系数，失败值为空。
- `eda_spectra.csv`：view/frequency/period_points/power；频率单位为每采样点，周期单位为点。
- `eda_outliers.csv`：time/value/method/tested_value/lower/upper；同一点被不同方法标记可有多行。
- `timings_seconds` 是分项计算耗时，不含绘图和序列化；不承诺新增分析后的总耗时必然下降。

## 预测导向证据

- 周期候选：配置周期、去趋势 ACF 峰值，以及固定频率可整除的日/周周期；月等日历频率不换算固定小时。
- 分成两个连续半段，各自线性去趋势，互相验证周期槽位均值；每半段至少覆盖两周期。
- 双向 R²≥0.1 且剖面相关≥0.5 才标记稳定候选；近零方差、样本不足标明 insufficient。
- 优先推荐通过筛选的配置周期，否则取通过筛选的最短周期；没有证据就不推荐，不回退到 FFT 全样本长度。
- 筛选使用全样本提出候选，是探索性证据，不是独立留出测试，也不是显著性检验。
- 差分建议至少需两个有效且方向一致的平稳性检验；差分后复检单独记录，不将建议当作已验证平稳。
- CH/OCSB 失败保留错误，D 为 null；D 只适用于实际接受检验的配置周期。
- 全局 IQR/Z-score 与 STL 残差 MAD（3×稳健标准差）并存；标记不等于数据错误，不自动修复。
- 趋势／季节强度不表示未来准确率；近零分量不能用分母加常数制造强度 1。

## BDS 策略

| AppConfig 参数 | 语义 |
| --- | --- |
| `eda_bds_mode=full` | 默认：全样本，不静默截断 |
| `eda_bds_mode=tail` | 显式取末尾连续窗口；只代表该窗口 |
| `eda_bds_mode=off` | 显式关闭，状态 skipped |
| `eda_bds_max_samples=0` | 默认：不设规模上限；tail 不允许 0，需 ≥10 |
| full + 正数上限 | 超限不调用 BDS，状态 resource_limit，不产生检验结论 |

BDS 样本对计算需要平方级内存；上限约束样本规模，不是操作系统内存/超时配额。报告披露模式、输入与实际样本数、起止时间和状态。默认全量行为不变。

## 状态与模型边界

- diagnostics 的 object 标识检验对象；status 区分 ok/failed/insufficient/skipped/resource_limit。
- 单项诊断容错，运行完成不等于每项检验完成；失败/跳过在报告 §10 汇总，manifest 仍负责产物完成与完整性。
- ARCH-LM 使用一阶差分，White/BP 使用线性时间趋势残差，BDS 使用原序列指定范围，均非预测模型残差。
- 报告 §9 默认只给参数约束；eda_task_confirmed=true 才基于已确认 history/horizon 生成共用窗口的候选 YAML，仍非最优参数。
- 建议不改写配置，报告不运行模型；全量分解不供回测训练复用，各训练窗口仍独立拟合。

返回 [EDA](eda.md) · [报告指南](eda_report_guide.md)。
