# 统计模型时间序列预测框架

本项目用于基于统计模型的时间序列预测，统一支持训练、回测、预测与 EDA（探索性数据分析）。

## 项目结构

```text
.
|- app/                # 应用编排层（pipeline/training/testing/forecasting）
|- config/             # dataclass 配置与 YAML/env 加载（default.py / loader.py）
|- models/             # 统计模型抽象、工厂与实现（`models/model/` 为分族包结构）
|- evaluation/         # 指标、滚动回测与可视化
|- data_provider/      # 数据聚合、加载、通用清洗与可逆预处理
|- features/           # 分析特征快照（feature_engineering / feature_scaling）
|- eda/                # EDA 子系统（诊断、建议、通用可视化与报告）
|- utils/              # demo 数据、运行时环境、日志、随机种子
|- scripts/            # 数据集专项运行脚本
|- run.py              # 唯一 CLI 入口
|- AGENTS.md           # 项目协作规范
|- CLAUDE.md           # Claude Code 工作指令
|- LOG.md              # 问题台账、修复记录与待办
```

> `dataset/`（数据集）、`logs/`、`results/`（运行输出）、`tests/`、`.venv/` 均在 `.gitignore` 中，属于本地目录，不会进入版本库；全新 clone 后需自行准备数据集并运行生成输出。

## 安装

```bash
uv venv .venv --python 3.12
uv sync --extra dev
```

当前项目统一使用根目录 `.venv` 的 `uv` 虚拟环境。新增或更新依赖时，统一使用 `uv add`，不要直接用 `pip install` 维护项目依赖。
为避免受限环境下的 `matplotlib` 缓存告警，项目默认使用根目录 `.mplconfig/` 作为本地 `MPLCONFIGDIR`。

常用依赖管理命令：

```bash
uv add <package>
uv add --dev <package>
uv sync --extra dev
```

## 运行示例

完整流程（训练 + 回测 + 预测）：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name arima --forecast_strategy direct --do_train true --do_test true --do_forecast true
```

仅执行 EDA：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py --do_eda true --do_train false --do_test false --do_forecast false
```

当前已接入的数据项目应优先使用独立 EDA 脚本：

```bash
bash scripts/wind_univariate/run_eda.sh
bash scripts/aidc_power_month/A/run_eda.sh
bash scripts/aidc_power_month/B/run_eda.sh
```

推荐工作流是：每份数据先运行一次对应的 `run_eda.sh`，阅读 `results/{data_name}/results_eda/` 中的诊断与建模建议，再运行任意数量的模型脚本。模型脚本统一保持 `--do_eda false`，不会重复执行数据分析。AIDC A/B 脚本分别从各自 5 分钟原始数据生成或复用日峰派生数据，并独立分析，不自动生成 A/B 比较图。

每次 `run_eda.sh` 还会自动生成一份中文叙述报告 `results/{data_name}/results_eda/.../EDA_REPORT.md`（由 `eda/report_generator.py` 渲染，复用 `eda_recommendations.json` 不重算阈值，含数据口径、趋势、季节性、平稳性、波动、建模建议等 8 段）。手写报告默认不会被覆盖（识别 `auto-generated` marker），需要刷新时加 `--eda_report_overwrite true`；精修参考 `eda/EDA_REPORT_GUIDE.md`。

执行 EDA 并输出建模建议：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py \
  --do_eda true --do_train false --do_test false --do_forecast false \
  --eda_period 7 --eda_nlags 24 --eda_recommendation_enabled true
```

运行前聚合高频数据（示例：5 分钟负荷聚合为日峰值）：

```bash
python -u run.py \
  --data_path dataset/aidc_power_month/A_Loads_5min_20251001_20260708.csv \
  --time_col time --target_col value --freq D \
  --aggregation_enabled true --aggregation_source_freq 5min \
  --aggregation_method max --aggregation_fill_method seasonal_slot \
  --aggregation_fill_weeks 4 \
  --aggregation_output_path dataset/aidc_power_month/derived/A_Loads_1day_20251001_20260708.csv
```

聚合发生在 DataLoader、EDA 和建模之前；派生 CSV 旁会生成 `.aggregate.json` 审计文件。默认不填充缺失时间点，支持显式选择 `linear` 或 `seasonal_slot`。

EDA 多序列比较：

```bash
python -u run.py \
  --data_path dataset/aidc_power_month/derived/A_Loads_1day_20251001_20260708.csv \
  --time_col time --target_col value --freq D \
  --do_eda true --do_train false --do_test false --do_forecast false \
  --eda_comparison_paths dataset/aidc_power_month/derived/B_Loads_1day_20251001_20260708.csv \
  --eda_comparison_labels B
```

启用预处理（可选去噪 + 去趋势 + 逆变换）：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py --denoise_method moving_median --denoise_window 5 --detrend_method linear
```

启用 ETS 调参与季节指数平滑：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py \
  --model_name ets \
  --model_params '{"trend":"add","seasonal":"add"}' \
  --seasonal_period 5 \
  --ets_tune_smoothing_params true \
  --ets_smoothing_grid_level 0.2,0.5,0.8 \
  --ets_smoothing_grid_trend 0.2,0.5 \
  --ets_smoothing_grid_seasonal 0.2,0.5 \
  --predict_horizon 4 \
  --do_train false --do_test false --do_forecast true
```

启用 ARIMA 家族分解预处理（自动周期推断 + 趋势/季节重组）：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py \
  --model_name ar \
  --model_params '{"p":2}' \
  --decomposition_method seasonal_decompose \
  --decomposition_target resid_only \
  --predict_horizon 4
```

启用预处理后 EDA 对比：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py \
  --do_eda true --do_train false --do_test false --do_forecast false \
  --decomposition_method seasonal_decompose \
  --decomposition_target resid_only \
  --seasonal_period 7 \
  --eda_run_preprocessed true
```

启用并行回测和本地监控：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py \
  --model_name naive \
  --do_train true --do_test true --do_forecast true \
  --backtest_n_jobs 2 \
  --monitor_enabled true
```

回填监控实际值并生成指标快照：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py \
  --monitor_actuals_path /abs/path/actuals.csv \
  --monitor_actuals_experiment_path naive-direct/params-default/... \
  --monitor_actuals_value_col actual \
  --monitor_actuals_run_id manual-backfill-1
```

多源输入预测（历史内生 + 历史外生 + 独立未来外生）：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py \
  --data_path /abs/path/history.csv \
  --time_col ds \
  --target_col y \
  --model_name linear_var \
  --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' \
  --endog_cols load \
  --exog_cols temp \
  --future_exog_path /abs/path/future_exog.csv \
  --future_exog_time_col ds \
  --future_exog_cols temp \
  --do_train true --do_test true --do_forecast true \
  --history_size 12 --predict_horizon 4
```

## 输出目录

结果统一使用 `--results_dir results`，先按派生/原始 `data_name` 分组，再按完整可读参数构建模型实验路径：

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

`experiment_path` 依次编码模型/策略、模型参数、history/predict、回测窗口、特征输入、预处理和预测区间。EDA 不包含模型名，按频率、周期、nlags、建议开关和聚合语义独立分组。`run_summary.json` 保存完整配置和所有产物路径。

## 当前主线说明

- 统计预测主线统一通过 `fit(y, X_hist=None, X_future=None) / predict_one(X_future_one=None)` 接口接入；多步推理统一由 `forecast_strategy` 编排。
- `forecast_strategy` 当前支持 `single_step / direct / recursive / dirrec`。
- `test` 额外支持 `backtest_window_mode = expanding | sliding`，并通过 `backtest_train_size` 控制训练窗口长度；旧 `backtest_initial_train_size` 仍兼容。`backtest_n_jobs > 1` 时可按窗口并行回测，输出仍按 `window_id` 排序。
- 当前主线默认仍输出单目标 `target_col -> yhat`，但已支持多源输入：
  - 历史内生协变量：`endog_cols`，不包含 `target_col`
  - 历史外生变量：`exog_cols`
  - 独立未来外生文件：`future_exog_path` + `future_exog_cols`
- 统计模型实现按 `models/model/` 家族模块维护，包含 ARIMA、指数平滑、多变量、波动率和扩展模型。
- 当前主线额外补充了一批轻量统计基线：
  - 基线：`seasonal_naive / historic_average`
  - 间歇需求：`croston`
  - 基于 `statsforecast` 的可选自动模型：`dynamic_theta / auto_ets / auto_theta`
- ARIMA 家族当前包含 `ar / ma / arma / arima / sarima / auto_arima`，统一由 `models/model/arima_family.py` 维护；模型 registry 位于 `models/registry.py`。
- 早期 ARIMA 研究脚本的有效内容已抽象为主线规则（脚本本身已移除，不再保留脚本式 API）：
  - `ACF/PACF` 用于 AR/MA/ARMA 识别与参数经验
  - `ADF/KPSS` 用于平稳性与差分建议
  - 分解观察与滚动预测思路用于主线测试和预处理设计
- `features/` 默认只用于生成分析型特征快照；当 `feature_mode=model_input` 时，会把时间特征和 lag 特征并入模型历史输入元信息，默认仍不改变单目标 `yhat` 输出 contract。
- 训练、测试、预测和监控按 `results/{data_name}/{category}/{experiment_path}` 分组；EDA 按独立的 `{eda_path}` 分组。
- 可选频率聚合由 `data_provider/data_aggregate.py` 在 DataLoader 前执行；`DataProcessor` 仍只负责可逆预处理。
- EDA 当前会输出结构化建模建议，包括季节周期、差分、预处理和模型族候选；启用 `eda_run_preprocessed` 后，会额外对实际训练尺度序列运行一轮 EDA。
- 测试阶段当前会额外输出窗口级明细、汇总指标和三类图：预测对比图、残差图、误差分布图。
- 无 `data_path` 时会加载内置 demo 序列，用于 smoke/test 场景；真实数据读取仍统一走 `data_provider/data_loader.py`。
- 通用时序清洗已统一收敛到 `data_provider.prepare_standard_frame()`：默认保留 `time_col/target_col`；配置多源列后会额外保留并数值化这些输入列。
- `data_provider/data_processor.py` 当前支持三层可逆处理：
  - 去噪：`moving_average / moving_median`
  - 去趋势：`linear / moving_average`
  - 分解：`seasonal_decompose / stl`，并支持 `trend_resid / resid_only` 两种建模目标
- `ETSModel` 当前继续作为统一入口承载 `SES / DES / TES(Holt-Winters)`：
  - `trend=None, seasonal=None` 对应 `SES`
  - `trend!=None, seasonal=None` 对应 `DES`
  - `trend!=None, seasonal!=None` 对应 `TES`
- `ETS` 的 `seasonal_periods` 来源统一为：
  - `model_params.seasonal_periods` 显式指定优先
  - 否则复用 `--seasonal_period`
  - 若仍缺失且模型启用了 `seasonal`，会尝试自动周期推断；推断失败则报错，不静默退化
- `LOWESS / Kalman` 当前不纳入主线预处理：
  - 依赖额外第三方库
  - 对当前统计预测主线增益有限，维护成本更高
- `dataset/wind_dataset.csv` 的单变量脚本已落在 `scripts/wind_univariate/`，当前约定 `DATE` 为时间列、`WIND` 为目标列。
- `var / bayesian_var / linear_var` 已接入主线；其中 `bayesian_var / linear_var` 当前标记为实验性多变量模型。
- `prophet / tbats / neuralprophet` 当前都已接入统一 `fit/predict` 契约：
  - `prophet` 支持 `growth`、`seasonality_mode`、`country_holidays` 与 future regressors
  - `tbats` 暴露多重季节性核心参数，如 `seasonal_periods`、`use_box_cox`、`use_trend`
  - `neuralprophet` 为实验性 optional 模型；若环境依赖不可用或运行时不兼容，会显式 fallback，不静默伪装成真实成功
- `bayesian_tmt` 当前明确表示“实验性单序列贝叶斯滞后回归近似”，不是旧 `BayesianTMT.py` 中的矩阵分解/面板预测算法。
- 模型稳定性约定（共 25 个模型，7 个家族）：
  - `stable`（12）：默认主线模型，可作为常规 baseline，如 `naive / ar / arima / sarima / ets / theta / var`
  - `optional`（7）：依赖额外库或环境能力，如 `dynamic_theta / auto_ets / auto_theta / arch / garch / tbats / prophet`
  - `experimental`（6）：实现已接入主线，但算法边界或环境稳定性仍需额外验证，如 `croston / bayesian_var / linear_var / neuralprophet / bayesian_tmt / rar`
- `models.stability.build_smoke_matrix()` 可生成 optional/experimental 模型 smoke matrix，状态固定为 `success / dependency_unavailable / fit_failed`，用于区分依赖问题、拟合失败和成功路径。

## 数据集脚本

- `dataset/`（本地、已 gitignore）当前包含 `wind_dataset.csv` 与若干公开基准：`ETT-small/`（ETTh1/h2、ETTm1/m2）、`weather/`、`electricity/`。目前只有 wind 单变量配有运行脚本，其余数据集建议直接用 CLI 运行。

- `scripts/wind_univariate/` 覆盖当前单变量可运行模型：
  - stable：`naive`、`seasonal_naive`、`historic_average`、`ar`、`ma`、`arma`、`arima`、`auto_arima`、`sarima`、`ets`、`theta`
  - optional：`dynamic_theta`、`auto_ets`、`auto_theta`、`arch`、`garch`、`tbats`、`prophet`
  - experimental：`croston`、`neuralprophet`、`bayesian_tmt`、`rar`
- 单模型脚本入口示例：
  - `bash scripts/wind_univariate/run_naive.sh`
  - `bash scripts/wind_univariate/run_seasonal_naive.sh`
  - `bash scripts/wind_univariate/run_arima.sh`
  - 其余模型同名脚本位于 `scripts/wind_univariate/`
- 当前每个脚本都显式传入 wind 单变量实验相关的主要 `AppConfig` 字段，并统一使用 `python -u run.py`、`model_name`、`LOG_NAME`
- 脚本中不展开 `config/config_module/config_class`、多源输入和 monitor actuals 回填字段；这些不属于 wind 单变量单模型运行入口。
- 脚本运行后会自动把结果写到 `results/{data_name}/` 下的对应类别与完整参数路径。
- 周期类日频脚本默认使用周周期参数，如 `season_length=7`、`period=7` 或 `seasonal_periods=7`；StatsForecast optional 模型显式传入 `freq="D"`。
- `run_auto_arima.sh` 当前定位为偏快的日常脚本：默认缩短 `history_size`、增大 `backtest_step`、收紧 `auto_arima` 搜索空间，并开启回测进度日志与 `auto_arima` trace
- `run_sarima.sh` 当前也定位为偏快的日常脚本：默认缩短 `history_size`、增大 `backtest_step`、开启回测进度日志，并通过 `model_params.fit_kwargs.maxiter` 等参数收紧 `SARIMAX` 拟合成本
- `run_tbats.sh` 与 `run_neuralprophet.sh` 保留为可运行脚本；当前环境 smoke 可能通过 fallback 完成，需结合 `test_summary.json` 或 smoke matrix 的 `used_fallback` 字段判断是否原生模型成功。
- 当前 `scripts/wind_univariate/` 只覆盖单变量脚本；多变量/多源输入模型建议直接用 CLI 运行。
- `scripts/wind_univariate/run_eda.sh` 与 `scripts/aidc_power_month/A|B/run_eda.sh` 是数据项目级 EDA 入口；模型脚本只负责训练、回测和预测，并统一关闭 EDA。
- `scripts/aidc_power_month/A|B/` 各包含22个日频模型脚本；脚本从只读5分钟原始文件生成/复用 `dataset/aidc_power_month/derived/` 日峰 CSV，再进入统一建模主线。

## EDA 能力

- 平稳性：ADF / KPSS / PP
- 分解：STL（趋势/季节/残差强度）
- 周期：FFT 主周期 + ACF 峰值候选周期
- 季节差分建议：CH / OCSB
- 异方差：ARCH-LM
- 白噪声：Ljung-Box
- 可预测性评分：谱熵归一化分数
- 建模建议：`eda_recommendations.json/csv`，覆盖季节周期、差分、预处理和模型族候选

## 监控闭环

- `monitor_enabled=true` 时，预测阶段会把每个未来步的 `yhat` 写入 `results/{data_name}/monitor/{experiment_path}/predictions_log.csv`。
- `evaluation.monitor.ModelMonitor` 和 `run.py --monitor_actuals_path ...` 支持后续回填真实值到 `actuals_log.csv`，并基于最近 `monitor_window` 个匹配样本生成 `metrics_history.csv`。
- 当前监控是本地文件版，不依赖数据库或服务端组件。

## 当前已知问题

- 若 `.venv` 与 `pyproject.toml` / `uv.lock` 不一致，需要重新执行 `uv sync --extra dev`。
- ARIMA 家族的高噪声初始化 warning 已在模型层定向过滤；当前仍可能看到少量 `ConvergenceWarning`。
- 自动季节周期推断默认优先使用 ACF 峰值，再回退到简单频域候选；若序列季节性不明显，会退回无季节路径。

详细问题与修复进度请见 `LOG.md`。

## 验证

```bash
UV_CACHE_DIR=.uv_cache uv run pytest -q
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name ets --model_params '{"trend":"add","seasonal":"add"}' --seasonal_period 5 --denoise_method moving_median --denoise_window 3 --decomposition_method seasonal_decompose --decomposition_target trend_resid --do_train false --do_test false --do_forecast true --history_size 60 --predict_horizon 4
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name ar --model_params '{"p":2}' --decomposition_method seasonal_decompose --decomposition_target resid_only --do_train false --do_test false --do_forecast true --history_size 60 --predict_horizon 4
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name seasonal_naive --model_params '{"season_length":7}' --do_train false --do_test false --do_forecast true --history_size 60 --predict_horizon 4
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name croston --model_params '{"alpha":0.2}' --do_train false --do_test false --do_forecast true --history_size 60 --predict_horizon 4
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name auto_theta --model_params '{"season_length":1}' --do_train false --do_test false --do_forecast true --history_size 60 --predict_horizon 4
UV_CACHE_DIR=.uv_cache uv run python run.py --do_eda true --do_train false --do_test false --do_forecast false
UV_CACHE_DIR=.uv_cache uv run python run.py --do_eda true --do_train false --do_test false --do_forecast false --eda_period 7 --eda_nlags 24 --eda_recommendation_enabled true
bash scripts/wind_univariate/run_eda.sh
bash scripts/aidc_power_month/A/run_eda.sh
bash scripts/aidc_power_month/B/run_eda.sh
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5 --backtest_n_jobs 2 --monitor_enabled true
UV_CACHE_DIR=.uv_cache uv run python run.py --monitor_actuals_path /abs/path/actuals.csv --monitor_actuals_experiment_path naive-direct/params-default/... --monitor_actuals_value_col actual --monitor_actuals_run_id manual-backfill-1
UV_CACHE_DIR=.uv_cache uv run python run.py --data_path /abs/path/history.csv --time_col ds --target_col y --model_name linear_var --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' --endog_cols load --exog_cols temp --future_exog_path /abs/path/future_exog.csv --future_exog_time_col ds --future_exog_cols temp --do_train true --do_test true --do_forecast true --history_size 12 --predict_horizon 4
```
