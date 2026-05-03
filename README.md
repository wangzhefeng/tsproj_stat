# 统计模型时间序列预测框架

本项目用于基于统计模型的时间序列预测，统一支持训练、回测、预测与 EDA（探索性数据分析）。

## 项目结构

```text
.
|- app/                # 应用编排层（pipeline/training/testing/forecasting）
|- config/             # dataclass 配置
|- models/             # 统计模型抽象、工厂与实现（`models/model/` 为分族包结构）
|- evaluation/         # 指标与滚动回测
|- data_provider/      # 数据加载、示例数据与预处理
|- features/           # 分析特征快照与后续扩展预留层
|- eda/                # EDA 子系统（analyzer/diagnostics/report/data_gen）
|- tests/              # 测试用例
|- scripts/            # 数据集专项运行脚本
|- saved_results/      # 运行期输出目录（按需生成）
|- run.py              # 完整 CLI 入口
|- main.py             # 最小示例入口
|- AGENTS.md           # 项目协作规范
|- LOG.md              # 问题台账、修复记录与待办
```

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
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name arima --pred_method direct --do_train true --do_test true --do_forecast true
```

仅执行 EDA：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py --do_eda true --do_train false --do_test false --do_forecast false
```

启用预处理（去噪 + 去趋势 + 逆变换）：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py --denoise_enabled true --denoise_window 5 --detrend_method linear
```

多源输入预测（历史内生 + 历史外生 + 独立未来外生）：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py \
  --data_path /abs/path/history.csv \
  --time_col ds \
  --target_col y \
  --model_name linear_var \
  --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' \
  --endog_cols y,load \
  --hist_exog_cols temp \
  --future_exog_path /abs/path/future_exog.csv \
  --future_exog_time_col ds \
  --future_exog_cols temp \
  --do_train true --do_test true --do_forecast true \
  --history_size 12 --predict_horizon 4
```

## 输出目录

当前结果统一按 `setting = {model_name}-{data_name}-{pred_method}` 落盘，例如 `arima-wind_dataset-direct`。

- `saved_results/checkpoints/{setting}/model.pkl`
- `saved_results/results_train/{setting}/train_summary.json`
- `saved_results/results_train/{setting}/train_series.csv`
- `saved_results/results_train/{setting}/model_info.json`
- `saved_results/results_eda/{setting}/eda_summary.json`
- `saved_results/results_eda/{setting}/eda_diagnostics.csv`
- `saved_results/results_eda/{setting}/plots/*.png`
- `saved_results/results_test/{setting}/backtest_predictions.csv`
- `saved_results/results_test/{setting}/backtest_metrics.csv`
- `saved_results/results_test/{setting}/backtest_metrics_summary.csv`
- `saved_results/results_test/{setting}/backtest_prediction_plot.png`
- `saved_results/results_test/{setting}/backtest_residual_plot.png`
- `saved_results/results_test/{setting}/backtest_error_distribution.png`
- `saved_results/results_forecast/{setting}/forecast.csv`
- `saved_results/results_forecast/{setting}/forecast_summary.json`
- `saved_results/results_forecast/{setting}/forecast_plot.png`
- `saved_results/results_forecast/{setting}/analysis_feature_snapshot.csv`
- `saved_results/results_forecast/{setting}/run_summary.json`

## 当前主线说明

- 统计预测主线统一通过 `fit(y, X_hist=None, X_future=None) / predict(horizon, X_future=None)` 接口接入。
- 当前主线默认仍输出单目标 `target_col -> yhat`，但已支持多源输入：
  - 内生变量：`endog_cols`
  - 历史外生变量：`hist_exog_cols`
  - 独立未来外生文件：`future_exog_path` + `future_exog_cols`
- 统计模型实现按 `models/model/` 家族模块维护，包含 ARIMA、指数平滑、多变量、波动率和扩展模型。
- ARIMA 选型 helper 位于 `models/model/arima_family.py`；模型 registry 位于 `models/registry.py`。
- `features/` 当前只用于生成分析型特征快照，不参与模型训练或预测主链路。
- 训练、测试、预测和 EDA 结果现统一落到 `saved_results/checkpoints / results_train / results_test / results_forecast / results_eda`，并按 `setting` 自动分组。
- 测试阶段当前会额外输出窗口级明细、汇总指标和三类图：预测对比图、残差图、误差分布图。
- 无 `data_path` 时会加载内置 demo 序列，用于 smoke/test 场景；真实数据读取仍统一走 `data_provider/data_loader.py`。
- 通用时序清洗已统一收敛到 `data_provider.prepare_standard_frame()`：默认保留 `time_col/target_col`；配置多源列后会额外保留并数值化这些输入列。
- `dataset/wind_dataset.csv` 的单变量脚本已落在 `scripts/wind_univariate/`，当前约定 `DATE` 为时间列、`WIND` 为目标列。
- `var / bayesian_var / linear_var` 已接入主线；其中 `bayesian_var / linear_var` 当前标记为实验性多变量模型。

## 数据集脚本

- 单模型脚本入口：
  - `bash scripts/wind_univariate/run_naive.sh`
  - `bash scripts/wind_univariate/run_arima.sh`
  - 其余模型同名脚本位于 `scripts/wind_univariate/`
- 当前每个脚本都显式传入完整 `AppConfig` 字段，并统一使用 `python -u run.py`、`model_name`、`LOG_NAME`
- 脚本运行后会自动把结果写到 `saved_results/checkpoints|results_train|results_test|results_forecast|results_eda/{setting}/`
- `run_auto_arima.sh` 当前定位为偏快的日常脚本：默认缩短 `history_size`、增大 `backtest_step`、收紧 `auto_arima` 搜索空间，并开启回测进度日志与 `auto_arima` trace
- `run_sarima.sh` 当前也定位为偏快的日常脚本：默认缩短 `history_size`、增大 `backtest_step`、开启回测进度日志，并通过 `model_params.fit_kwargs.maxiter` 等参数收紧 `SARIMAX` 拟合成本
- 当前 `scripts/wind_univariate/` 只覆盖单变量脚本；多变量/多源输入模型建议直接用 CLI 运行。

## EDA 能力

- 平稳性：ADF / KPSS / PP
- 分解：STL（趋势/季节/残差强度）
- 周期：FFT 主周期 + ACF 峰值候选周期
- 季节差分建议：CH / OCSB
- 异方差：ARCH-LM
- 白噪声：Ljung-Box
- 可预测性评分：谱熵归一化分数

## 数据生成脚本

- `eda/data_gen.py`：用于生成模型测试数据。
- 默认输出路径需以脚本实际配置为准；若新增数据目录，需同步更新 `AGENTS.md` 与本文件。

## 迁移说明

- `todo_ts_eda` 中“趋势去除与逆变换”“去噪流程”已并入 `data_provider/data_processor.py`。
- `todo_ts_eda` 其余主要统计诊断能力已并入 `eda/diagnostics.py` 与 `eda/report.py`。
- `todo_models_source` 中 `BayesianTMT` 与 `RAR` 已完成非占位迁移。
- `todo_models_source/` 与 `todo_ts_eda/` 目录已删除。

## 当前已知问题

- 若 `.venv` 与 `pyproject.toml` / `uv.lock` 不一致，需要重新执行 `uv sync --extra dev`。
- ARIMA 家族的高噪声初始化 warning 已在模型层定向过滤；当前仍可能看到少量 `ConvergenceWarning`。

详细问题与修复进度请见 `LOG.md`。

## 验证

```bash
UV_CACHE_DIR=.uv_cache uv run pytest -q
UV_CACHE_DIR=.uv_cache uv run python run.py --model_name naive --do_train true --do_test true --do_forecast true --history_size 60 --predict_horizon 5
UV_CACHE_DIR=.uv_cache uv run python run.py --do_eda true --do_train false --do_test false --do_forecast false
UV_CACHE_DIR=.uv_cache uv run python run.py --data_path /abs/path/history.csv --time_col ds --target_col y --model_name linear_var --model_params '{"target_lags":[1,2],"feature_lags":[0,1]}' --endog_cols y,load --hist_exog_cols temp --future_exog_path /abs/path/future_exog.csv --future_exog_time_col ds --future_exog_cols temp --do_train true --do_test true --do_forecast true --history_size 12 --predict_horizon 4
```
