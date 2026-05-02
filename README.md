# 统计模型时间序列预测框架

本项目用于基于统计模型的时间序列预测，统一支持训练、回测、预测与 EDA（探索性数据分析）。

## 项目结构

```text
.
|- app/                # 应用编排层（pipeline/training/testing/forecasting）
|- config/             # dataclass 配置
|- models/             # 统计模型抽象、工厂与实现（`models/statistical/` 为分族包结构）
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
UV_CACHE_DIR=.uv_cache uv run python run.py --model-name arima --pred-method direct --do-train true --do-test true --do-forecast true
```

仅执行 EDA：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py --do-eda true --do-train false --do-test false --do-forecast false
```

启用预处理（去噪 + 去趋势 + 逆变换）：

```bash
UV_CACHE_DIR=.uv_cache uv run python run.py --denoise-enabled true --denoise-window 5 --detrend-method linear
```

## 输出目录

当前结果统一按 `setting = {model_name}-{data_name}-{pred_method}` 落盘，例如 `arima-wind_dataset-direct`。

- `saved_results/checkpoints/{setting}/model.pkl`
- `saved_results/results_train/{setting}/train_summary.json`
- `saved_results/results_train/{setting}/train_series.csv`
- `saved_results/results_train/{setting}/model_info.json`
- `saved_results/results_train/{setting}/eda/eda_summary.json`
- `saved_results/results_train/{setting}/eda/eda_diagnostics.csv`
- `saved_results/results_train/{setting}/eda/plots/*.png`
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

- 统计预测主线当前仍是单变量序列建模，统一通过 `fit / predict` 接口接入。
- 统计模型实现已从单文件 `models/statistical.py` 重构为 `models/statistical/` 包，按 ARIMA、指数平滑、多变量、波动率和扩展模型分组维护。
- ARIMA 选型 helper 已并入 `models/statistical/arima_family.py`；模型 registry 当前位于 `models/registry.py`，由 `models.statistical` 做兼容导出。
- `features/` 当前只用于生成分析型特征快照，不参与模型训练或预测主链路。
- 训练、测试、预测和 EDA 结果现统一落到 `saved_results/` 的四类一级目录中，并按 `setting` 自动分组。
- 测试阶段当前会额外输出窗口级明细、汇总指标和三类图：预测对比图、残差图、误差分布图。
- 无 `data_path` 时会加载内置 demo 序列，用于 smoke/test 场景；真实数据读取仍统一走 `data_provider/data_loader.py`。
- `dataset/wind_dataset.csv` 的单变量脚本已落在 `scripts/wind_univariate/`，当前约定 `DATE` 为时间列、`WIND` 为目标列。

## 数据集脚本

- 单模型脚本入口：
  - `bash scripts/wind_univariate/run_naive.sh`
  - `bash scripts/wind_univariate/run_arima.sh`
  - 其余模型同名脚本位于 `scripts/wind_univariate/`
- 当前每个脚本都直接写死 `dataset/wind_dataset.csv`、`DATE`、`WIND` 与输出目录参数，不依赖公共 `common.sh`
- 脚本运行后会自动把结果写到 `saved_results/checkpoints|results_train|results_test|results_forecast/{setting}/`
- 当前单变量脚本覆盖所有 `supports_multivariate=False` 的模型；`var / bayesian_var / linear_var` 暂未纳入。

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
UV_CACHE_DIR=.uv_cache uv run python run.py --model-name naive --do-train true --do-test true --do-forecast true --history-size 60 --predict-horizon 5
UV_CACHE_DIR=.uv_cache uv run python run.py --do-eda true --do-train false --do-test false --do-forecast false
```
