# 数据与聚合

## 数据集布局

`dataset/`（本地、已 gitignore）：

| 数据 | 位置 | 说明 |
| --- | --- | --- |
| wind | `wind/wind_dataset.csv` | 单变量风电，`DATE` 时间列、`WIND` 目标列 |
| AIDC 负荷 | `aidc_power_month/` | A/B 两路 5min 原始 + `derived/` 派生 + `plots/` |
| 公开基准 | `ETT-small/`、`weather/`、`electricity/` | 直接用 CLI 运行，无专项脚本 |

数据读取统一走 `data_provider/data_loader.py`；无 `data_path` 时加载内置 demo 序列（`utils/demo_data.py`），用于 smoke/test 场景。通用清洗收敛到 `data_provider.prepare_standard_frame()`。

多源输入（内生/外生/未来外生）约定见 [exogenous.md](exogenous.md)。

## 频率聚合

高频数据在 DataLoader、EDA 与建模之前聚合，由 `data_provider/data_aggregate.py` 执行：

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

## 数据项目脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/wind_univariate/run_eda.sh` + 22 个模型脚本 | wind 单变量全模型入口 |
| `scripts/aidc_power_month/route_A\|route_B/run_eda.sh` + 各 29 个日频脚本 | AIDC 两路：从只读 5min 原始聚合生成/复用 `derived/` 日频 CSV；结果落 `results/aidc_power_month/route_A\|route_B/`（`--results_data_name`），过期窗口的历史结果归档在 `results/aidc_power_month/_archive/` |
| `scripts/aidc_power_month/run_all.sh` | A/B 各 16 个基准配置串行批跑；日志落项目根 `logs/aidc_power_month/run_all_<时间戳>/` |

注意：删除审计 JSON 后重跑会触发重聚合并覆盖派生文件；有人工调整的派生数据重跑前先备份。

相关：[eda.md](eda.md)（EDA 输出）、[LOG.md](LOG.md)（数据维护记录）。
