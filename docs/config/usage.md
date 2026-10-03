# 运行与 CLI

建模入口 `run.py`；独立 EDA 入口 `run_eda.py` 调用 `eda/runner.py`，读取 scripts 下的场景 YAML。两者复用 `AppConfig` 字段与配置加载器，完整参数见 `config/default.py`。

## 配置来源

优先级从低到高：默认配置类 → `--config` YAML → `TSPROJ_<字段名大写>` 环境变量 → CLI 显式参数；未传 YAML 也读取环境变量。来源共用 `config/loader.py` 转换：bool 接受 `true/1/yes/y/on` 与 `false/0/no/n/off`；列表接受逗号分隔字符串或原生 list；dict 接受 JSON 对象文本。YAML 未登记字段报错，Optional 字段允许原生 `null`，必填字段拒绝 `null`。CLI 未提供的 None 不覆盖低优先级来源。
CLI 参数按 AppConfig 字段自动注册（int/float 标量直转、`list[float]` 用空格分隔多值、其余透传转换）。新增字段除在 `config/default.py` 定义外，还须在 `artifacts/identity.py` 分类登记，并实现运行消费与测试。

## 常用命令

完整流程（训练 + 回测 + 预测）：

```bash
.venv/bin/python run.py --model_name arima --forecast_strategy direct --do_train true --do_test true --do_forecast true
```

多模型单 run 对比（数据准备/EDA 只做一次，各模型独立 experiment_path）：

```bash
.venv/bin/python run.py --model_names naive,arima,seasonal_naive --auto_select true --do_train true --do_test true --do_forecast true
```

对比表写入 `results/{data_name}/results_test/comparison/runs/{run_id}/model_comparison.csv`；开启 `auto_select` 时它是独立尾段成绩，不能用于改选；指标规则见 [testing.md](../evaluation/testing.md)。`batch_models` 提供每模型独立参数；未覆盖的模型使用全局 `model_params`。
`model_names` 去重保序，未设置回退 `[model_name]`；`batch_models` 的 `{model: params}` 在 series_id_col 为空时用于单表多模型，非空时为面板任务参数表。
单/多模型 `auto_select` 先保留独立尾段，再在前段最多评估 `auto_select_n_windows` 个窗口；`auto_select_holdout_size` 默认保留至少一个 horizon 的 20% 尾段。单模型候选缺省 registry stable，多模型用 model_names；最终训练/未来预测仍用最新 history_size 行。选型、外生声明等行为变化见 [信息集](../pipeline/information-set.md)。

仅 EDA：

BDS 默认全量；`--eda_bds_mode tail --eda_bds_max_samples 2000` 显式检验末段，`off` 关闭；full 配正数上限时超限明确标记不执行，不静默截取。详见 [EDA 契约](../eda/evidence.md)。

```bash
.venv/bin/python run_eda.py --config scripts/aidc_power_month/route_A/eda/15min.yaml
```

train 阶段拟合值诊断（模型须声明 `supports_fitted_values`；产物与数值契约见 [拟合值诊断](../models/fitted-values.md)）：

```bash
.venv/bin/python run.py --model_name arima --train_fitted_values true --do_train true
```

EDA 场景归属、路径规则和入口门禁见 [scenarios.md](../eda/scenarios.md)；原 run.py --do_eda 接口保留兼容，新增场景使用专用入口。

预处理示例（去噪 + 去趋势）、ETS 调参、分解预处理组合等场景命令见 [preprocessing.md](../data_provider/preprocessing.md)；策略与区间命令见 [strategies.md](../forecasting/strategies.md)；多源输入命令见 [exogenous.md](../pipeline/exogenous.md)；监控回填命令见 [monitoring.md](../monitoring/monitoring.md)。

## 阶段开关

- `--do_train` / `--do_test` / `--do_forecast` / `--do_eda` 独立控制四个阶段
- `--do_train false --do_forecast true` 是合法的原点即时训练用法
- EDA 属数据项目级流程：模型 shell 必须显式 `--do_eda false`，且不携带其他 `--eda_*` 参数

## 输出目录

结果统一 `--results_dir results`；`--results_data_name` 支持合法相对层级，如 `aidc_power_month/route_A`。
新运行使用带配置摘要的实验路径和 `runs/{run_id}` 隔离；详细目录、manifest、归档与迁移约定见 [产物协议](../artifacts/artifacts.md)。
消费 `run_summary.json` 返回的路径，不拼接旧固定目录；存量结果不迁移、不删除。

- checkpoint 仅训练归档，forecast/test 按策略即时拟合，不消费 checkpoint。
- forecast 原点 = 数据末尾，history = 尾部 `history_size` 行；历史评估由 rolling backtest 承担（见 [testing.md](../evaluation/testing.md)）
