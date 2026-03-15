# 统计模型时间序列预测框架

本项目用于基于统计模型的时间序列预测，统一支持训练、回测、预测与 EDA（探索性数据分析）。

## 主线结构

```text
.
|- app/                # 应用编排层（pipeline/training/testing/forecasting）
|- config/             # dataclass 配置
|- models/             # 统计模型抽象、工厂与实现
|- evaluation/         # 指标与滚动回测
|- data_provider/      # 数据加载与预处理（data_loader/data_processor）
|- features/           # 特征工程与缩放
|- eda/                # EDA 子系统（analyzer/diagnostics/report）
|- datasets/           # 示例与模拟数据
|- run.py              # 完整 CLI 入口
|- main.py             # 最小示例入口
```

## 安装

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .
pip install -e .[dev]
```

## 运行

完整流程（训练 + 回测 + 预测）：

```bash
python run.py --model-name arima --pred-method direct --do-train true --do-test true --do-forecast true
```

仅执行 EDA：

```bash
python run.py --do-eda true --do-train false --do-test false --do-forecast false
```

启用预处理（去噪 + 去趋势 + 逆变换）：

```bash
python run.py --denoise-enabled true --denoise-window 5 --detrend-method linear
```

## 产物目录

- `artifacts/checkpoints/model.pkl`
- `artifacts/results_test/backtest_metrics.csv`
- `artifacts/results_forecast/prediction.csv`
- `artifacts/results_forecast/feature_snapshot.csv`
- `artifacts/results_forecast/run_summary.json`
- `artifacts/results_eda/eda_summary.json`
- `artifacts/results_eda/eda_diagnostics.csv`
- `artifacts/results_eda/plots/*.png`

## EDA 覆盖能力

- 平稳性：ADF / KPSS / PP
- 分解：STL（趋势/季节/残差强度）
- 周期：FFT 主周期 + ACF 峰值候选周期
- 季节差分建议：CH / OCSB
- 异方差：ARCH-LM
- 白噪声：Ljung-Box
- 可预测性评分：谱熵归一化分数

## 本次整合进展

- `todo_ts_eda` 中“趋势去除与逆变换”“去噪流程”已并入 `data_provider/data_processor.py`。
- `todo_ts_eda` 其余主要统计诊断能力已并入 `eda/diagnostics.py` 与 `eda/report.py`。
- `todo_models_source` 中 `BayesianTMT` 与 `RAR` 已完成非占位迁移，现为可训练可预测实现。
- `todo_models_source/` 与 `todo_ts_eda/` 目录已删除。

## 验证

```bash
pytest -q
python run.py --do-eda true --do-train false --do-test false --do-forecast false
```