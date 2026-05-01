# 统计模型时间序列预测框架

本项目用于基于统计模型的时间序列预测，统一支持训练、回测、预测与 EDA（探索性数据分析）。

## 项目结构

```text
.
|- app/                # 应用编排层（pipeline/training/testing/forecasting）
|- config/             # dataclass 配置
|- models/             # 统计模型抽象、工厂与实现
|- evaluation/         # 指标与滚动回测
|- data_provider/      # 数据加载与预处理（data_loader/data_processor）
|- features/           # 特征工程与缩放
|- eda/                # EDA 子系统（analyzer/diagnostics/report/data_gen）
|- tests/              # 测试用例
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

常用依赖管理命令：

```bash
uv add <package>
uv add --dev <package>
uv sync --extra dev
```

## 运行示例

完整流程（训练 + 回测 + 预测）：

```bash
uv run python run.py --model-name arima --pred-method direct --do-train true --do-test true --do-forecast true
```

仅执行 EDA：

```bash
uv run python run.py --do-eda true --do-train false --do-test false --do-forecast false
```

启用预处理（去噪 + 去趋势 + 逆变换）：

```bash
uv run python run.py --denoise-enabled true --denoise-window 5 --detrend-method linear
```

## 输出目录

- `saved_results/checkpoints/model.pkl`
- `saved_results/results_test/backtest_metrics.csv`
- `saved_results/results_forecast/prediction.csv`
- `saved_results/results_forecast/feature_snapshot.csv`
- `saved_results/results_forecast/run_summary.json`
- `saved_results/results_eda/eda_summary.json`
- `saved_results/results_eda/eda_diagnostics.csv`
- `saved_results/results_eda/plots/*.png`

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
- `src/ts_forecast_framework/` 仍有历史残留，暂未清理。
- 个别历史命名仍待统一，例如 `FeatureScalering.py`。

详细问题与修复进度请见 `LOG.md`。

## 验证

```bash
uv run pytest -q
uv run python run.py --model-name naive --do-train true --do-test true --do-forecast true --history-size 60 --predict-horizon 5
uv run python run.py --do-eda true --do-train false --do-test false --do-forecast false
```
