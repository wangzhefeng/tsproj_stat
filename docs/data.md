# 数据与聚合

## 数据集布局

`dataset/`（本地、已 gitignore）：

| 数据 | 位置 | 说明 |
| --- | --- | --- |
| wind | `wind/wind_dataset.csv` | 单变量风电，`DATE` 时间列、`WIND` 目标列 |
| AIDC 负荷 | `aidc_power_month/` | A/B 两路 5min 原始 + `derived/` 派生 + `plots/` |
| 公开基准 | `ETT-small/`、`weather/`、`electricity/` | 直接用 CLI 运行，无专项脚本 |

数据读取统一走 `data_provider/loading/loader.py`；无数据源时加载 demo（`utils/demo_data.py`）。职责及旧接口迁移见 [data-architecture.md](data-architecture.md)。

CSV、内存帧与 demo 共用 `cleaning/normalization.py` 和 `quality/`：只规范化时间/字段/数值，保留 NaN 和时间缺口，不删行、不插值。`max_missing_ratio` 在修复前检查目标；无时间列时仍保留 demo 兼容自动时间列。

`data_quality.json` 中 `missing_timestamp_count` 是缺口，`inserted_timestamp_count` 是实际插入（Loader 为零）；不再将缺失数量差冒充插值计数。窗口修复由 `cleaning/imputation.py` 返回实际按列计数，存入运行/训练摘要和回测窗口指标。

多源输入（内生/外生/未来外生）约定见 [exogenous.md](exogenous.md)。

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
- 派生 CSV 旁生成 `.aggregate.json` 审计（源指纹、频率、填充统计）；审计存在且参数一致时复用，否则重算覆盖
- `linear` 与 `seasonal_slot` 为双向填充（离线数据准备可接受，但审计中披露 `fill_uses_future`）
- 复用旧独立脚本缓存时，缺失的填充方向披露自动补入审计 JSON，不重算或改写 CSV

## 数据项目脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/aidc_power_month/prepare_data.py` | A/B 共享数据准备入口：5min → 15min/小时/日均值，复用通用算法；日期与路径等场景配置仅在此维护 |
| `scripts/wind_univariate/run_eda.sh` + 22 个模型脚本 | wind 单变量全模型入口 |
| `scripts/aidc_power_month/route_A\|route_B/run_eda.sh` + 各 29 个日频脚本 | AIDC 两路：从只读 5min 原始聚合生成/复用 `derived/` 日频 CSV；结果落 `results/aidc_power_month/route_A\|route_B/`（`--results_data_name`），过期窗口的历史结果归档在 `results/aidc_power_month/_archive/` |
| `scripts/aidc_power_month/run_all.sh` | A/B 各 16 个基准配置串行批跑；日志落项目根 `logs/aidc_power_month/run_all_<时间戳>/` |

注意：删除审计 JSON 后重跑会触发重聚合并覆盖派生文件；有人工调整的派生数据重跑前先备份。

独立准备：`.venv/bin/python scripts/aidc_power_month/prepare_data.py`。可用 `--data-dir`、`--output-dir`、`--date-range` 覆盖输入目录、输出目录与文件日期标识；默认输出仍为原数据目录下 `derived/`。支持绝对脚本路径从其他工作目录调用；MC/Hermes 加 `env -u PYTHONPATH` 前缀。

相关：[eda.md](eda.md)（EDA 输出）、[LOG.md](LOG.md)（数据维护记录）。
