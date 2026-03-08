# 统计模型时间序列预测测试与推理框架

一个基于统计模型（ARIMA / Naive）的时间序列预测骨架，包含训练、推理、回测、评估与模型持久化。

## 目录结构

```text
.
├── config/
│   └── default.yaml
├── examples/
│   └── run_backtest.py
├── src/
│   └── ts_forecast_framework/
│       ├── data.py
│       ├── persistence.py                # 模型保存/加载
│       ├── evaluation/
│       │   ├── backtest.py
│       │   └── metrics.py
│       ├── inference/
│       │   └── predict.py
│       └── models/
│           ├── base.py
│           ├── selection.py              # AIC/BIC 自动选阶
│           └── statistical.py            # Naive/ARIMA
├── tests/
│   ├── test_metrics.py
│   ├── test_backtest_smoke.py
│   ├── test_arima_smoke.py
│   ├── test_arima_auto_order.py
│   └── test_persistence.py
├── pyproject.toml
└── requirements.txt
```

## 已实现能力

- 统一模型接口：`fit / predict`
- ARIMA 真实训练与预测（`statsmodels`）
- ARIMA 自动选阶（网格搜索 + `AIC/BIC`）
- 回测评估（MAE / RMSE / MAPE）
- 模型保存与加载（pickle）

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python examples/run_backtest.py
```

## 示例：ARIMA 自动选阶

```python
from ts_forecast_framework.models.statistical import ARIMAForecaster

model = ARIMAForecaster(
    auto_order=True,
    order_grid=[(0,1,0), (1,1,0), (1,1,1)],
    ic="aic",
)
```

## 示例：模型持久化

```python
from ts_forecast_framework.persistence import save_model, load_model

save_model(model, "artifacts/model.pkl")
model2 = load_model("artifacts/model.pkl")
```
