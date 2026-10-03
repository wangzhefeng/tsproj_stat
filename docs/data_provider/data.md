# 数据与聚合

## 数据集布局

`dataset/`（本地、已 gitignore）：

| 数据 | 位置 | 说明 |
| --- | --- | --- |
| wind | `wind/wind_dataset.csv` | 单变量风电，`DATE` 时间列、`WIND` 目标列 |
| AIDC 负荷 | `aidc_power_month/` | A/B 两路 5min 原始 + `derived/` 派生 + `plots/` |
| 公开基准 | `ETT-small/`、`weather/`、`electricity/` | ETTm1 有专项脚本，其余可用 CLI |

数据读取统一走 `data_provider/loading/loader.py`；无数据源时加载 demo（`utils/demo_data.py`）。职责及旧接口迁移见 [data-architecture.md](data-architecture.md)。

CSV、内存帧与 demo 共用 `cleaning/normalization.py` 和 `quality/`：只规范化时间/字段/数值，保留 NaN 和时间缺口，不删行、不插值。`max_missing_ratio` 在修复前检查目标；无时间列时仍保留 demo 兼容自动时间列。

`data_quality.json` 中 `missing_timestamp_count` 是缺口，`inserted_timestamp_count` 是实际插入（Loader 为零），不以缺失数量差冒充插值计数。`cleaning/imputation.py` 仅在已切历史窗内线性修复，全缺列失败；实际按列计数存入运行/训练摘要及回测指标，评估真值不填。

逐列报告 `missing_by_column/raw_missing_by_column/coercion_failed_by_column/nonfinite_by_column` 区分规范化后缺失、原始缺失、数值化失败和无穷值。建模前拒绝空、无效、重复、不等频时间轴；`validate_freq=false` 仅关闭 Loader 告警，不绕过建模门禁。

多源输入（内生/外生/未来外生）约定见 [exogenous.md](../pipeline/exogenous.md)。

## 频率聚合

高频数据先聚合：`data_provider/resampling/core.py::aggregate_frame` 负责内存计算；`resampling/service.py` 负责 CSV、审计、缓存；`pipeline/data_preparation.py` 适配 AppConfig：

```bash
.venv/bin/python run.py \
  --data_path dataset/aidc_power_month/A_Loads_5min_20251001_20260728.csv \
  --time_col time --target_col value --freq D \
  --aggregation_enabled true --aggregation_source_freq 5min \
  --aggregation_method mean --aggregation_fill_method seasonal_slot \
  --aggregation_fill_weeks 4 \
  --aggregation_output_path dataset/aidc_power_month/derived/A_Loads_1day_mean_20251001_20260728.csv
```

- 方法：`mean / max / min / sum / median`；缺失策略：`none / linear / seasonal_slot`
- 聚合拒绝空/NaT/非有限目标和源网格错位；重复时间按均值合并并审计。固定步长仅支持整倍数降频/同频；日内细分及 D 可聚合到周/月/季/年桶，其他跨日历组合明确拒绝。边界不完整桶保留已有 pandas 语义；`sum` 不自动换算电量。
- 派生 CSV 旁生成 v2 `.aggregate.json`：参数、源 SHA-256、输出 SHA-256 和审计摘要均匹配才复用；旧版/缺字段/损坏均重建。人工修改 CSV 也会触发重算，需另存路径。
- `linear` 与 `seasonal_slot` 为双向填充（离线数据准备可接受，但审计中披露 `fill_uses_future`）
- 同目标进程锁（macOS/Linux）覆盖校验与发布；`.CSV文件名.lock` 空锁文件保留。唯一临时目录避免写入冲突；CSV/审计分两次替换，不是联合原子事务，中断不匹配会在下次调用重建。

## 数据项目脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/aidc_power_month/prepare_data.py` | A/B 共享数据准备入口：5min → 15min/小时/日均值，复用通用算法；日期与路径等场景配置仅在此维护 |
| `scripts/wind/eda/run_eda.sh` | wind 数据集独立 EDA，默认配置为同目录 D.yaml |
| `scripts/wind/univariate/` 26 个模型脚本 | wind 单变量入口；批量对比在 `scripts/wind/run_models_all.sh`（neuralprophet 保留单脚本但不入批量） |
| `scripts/wind/multivariate/` 3 个模型脚本 + run_models_all.sh | wind 多变量入口：WIND + RAIN/T.MAX/T.MIN/T.MIN.G 内生面板（var/bayesian_var/linear_var），缺失窗口内线性插值并逐列入审计，评估只看 WIND |
| `scripts/ett_small/ETTm1/` | eda + univariate + multivariate；OT 目标，多变量另读 6 列内生协变量；单变量批量入口在场景根 |
| `scripts/aidc_power_month/route_A\|route_B/eda/run_eda.sh` | 选择同目录 `*.yaml`，调用根 run_eda.py；默认日频，只读派生 CSV；可用 --config 切换粒度 |
| AIDC 各路 `univariate/` 日频模型脚本 | 从 5min 原始聚合生成/复用派生 CSV；模型结果仍落对应 route 子树；批量入口留 route 根 |
| `scripts/aidc_power_month/run_all.sh` | A/B 各 16 个基准配置串行批跑；日志落项目根 `logs/aidc_power_month/run_all_<时间戳>/` |

注意：数据准备/启用聚合的模型入口会在审计缺失时重聚合覆盖；独立 EDA 不会，只披露来源未验证。

独立准备：`.venv/bin/python scripts/aidc_power_month/prepare_data.py`。可用 `--data-dir`、`--output-dir`、`--date-range` 覆盖输入目录、输出目录与文件日期标识；默认输出仍为原数据目录下 `derived/`。支持绝对脚本路径从其他工作目录调用；MC/Hermes 加 `env -u PYTHONPATH` 前缀。

相关：[脚本组织](scenario-scripts.md)（AIDC 共用体、场景合并与 EDA 分离）、[EDA](../eda/eda.md)（分析输出）、[产物协议](../artifacts/artifacts.md)（运行结果与审计）。
