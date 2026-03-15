# 统计模型时间序列预测框架

本项目用于基于统计模型的时间序列预测，采用根目录主线分层架构，统一支持训练、回测、预测与模型持久化。

## 项目结构

```text
.
|- app/                # 应用编排层（pipeline / training / testing / forecasting）
|- config/             # dataclass 配置
|- models/             # 统计模型抽象、工厂与实现
|- evaluation/         # 指标与回测
|- inference/          # 预测策略分发（one_step / recursive / direct）
|- data_provider/      # 数据加载与窗口切分
|- features/           # 特征工程与缩放
|- datasets/           # 示例与模拟数据
|- run.py              # CLI 入口
|- main.py             # 本地最小入口
```

## 统计预测策略

- `one_step`：一步预测
- `recursive`：递归多步预测
- `direct`：直接多步预测
- `rolling_backtest`：滚动窗口回测（见 `evaluation/backtest.py`）

## 快速开始

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
```

## 运行示例

```bash
python run.py --model-name arima --pred-method direct --do-train true --do-test true --do-forecast true
```

## 输出目录

- `artifacts/checkpoints/model.pkl`
- `artifacts/results_test/backtest_metrics.csv`
- `artifacts/results_forecast/prediction.csv`
- `artifacts/results_forecast/feature_snapshot.csv`
- `artifacts/results_forecast/run_summary.json`

## 模拟数据

- `datasets/simulated_daily.csv`：用于本地快速冒烟验证。

## 本次更新（2026-03-15）

- `main.py` 与 `run.py` 新增常见 `statsmodels` 拟合警告的定向过滤：
  - `Non-invertible starting MA parameters found`
  - `ConvergenceWarning`
- 目的：减少噪声日志，保留主流程输出，不影响模型训练与预测逻辑。
