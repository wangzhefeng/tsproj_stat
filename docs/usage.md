# 运行与 CLI

统一入口为 `run.py`，CLI 参数名与 `AppConfig` 字段一致（如 `--model_name`、`--predict_horizon`、`--do_eda`）。完整参数见 `config/default.py`。

## 常用命令

完整流程（训练 + 回测 + 预测）：

```bash
.venv/bin/python run.py --model_name arima --forecast_strategy direct --do_train true --do_test true --do_forecast true
```

仅 EDA：

```bash
.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false
```

已接入数据项目优先用独立 EDA 脚本（见 [data.md](data.md#数据项目脚本)）。

预处理示例（去噪 + 去趋势）、ETS 调参、分解预处理组合等场景命令见 [preprocessing.md](preprocessing.md)；策略与区间命令见 [strategies.md](strategies.md)；多源输入命令见 [exogenous.md](exogenous.md)；监控回填命令见 [monitoring.md](monitoring.md)。

## 阶段开关

- `--do_train` / `--do_test` / `--do_forecast` / `--do_eda` 独立控制四个阶段
- `--do_train false --do_forecast true` 是合法的原点即时训练用法
- EDA 属数据项目级流程：模型 shell 必须显式 `--do_eda false`，且不携带其他 `--eda_*` 参数

## 输出目录

结果统一 `--results_dir results`，先按 `data_name` 分组，再按完整可读参数构建 `experiment_path`：

```text
results/{data_name}/
├── checkpoints/{experiment_path}/
├── custom_monitor/{experiment_path}/
├── monitor/{experiment_path}/
├── results_forecast/{experiment_path}/
├── results_test/{experiment_path}/
├── results_train/{experiment_path}/
└── results_eda/{eda_path}/
```

`data_name` 默认取 `data_path` 文件 stem；可用 `--results_data_name` 显式覆盖（支持层级路径），把同一数据项目的多路线结果组织到统一子树——如 AIDC 两路脚本传 `aidc_power_month/route_A|route_B`，结果落 `results/aidc_power_month/route_A/`，与 `scripts/aidc_power_month/` 的组织镜像对应。绝对路径与 `..` 上跳段会被拒绝。

`experiment_path` 依次编码模型/策略、模型参数、history/predict、回测窗口、特征输入、预处理和预测区间；EDA 不含模型名，按频率、周期、nlags 和聚合语义独立分组。`run_summary.json` 保存完整配置与全部产物路径。

- `checkpoints/` 下 `model.pkl`/`model_meta.json` 是 train 阶段归档产物；forecast/test 按策略在推理时拟合，不消费 checkpoint
- forecast 原点 = 数据末尾，history = 尾部 `history_size` 行；历史评估由 rolling backtest 承担（见 [testing.md](testing.md)）
