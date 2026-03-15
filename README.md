# 统计模型时间序列预测框架（重构版）

本项目已按 `tsproj_ml/stable` 风格完成一次性重构，采用分层架构：配置层、模型工厂层、训练/测试/预测编排层、CLI 入口层。

## 新架构

```text
.
├─ src/ts_forecast_framework/
│  ├─ config/                # dataclass 配置
│  ├─ models/                # 统一模型抽象 + 工厂
│  ├─ evaluation/            # 指标与回测
│  ├─ app/                   # Pipeline 编排
│  ├─ training.py            # Trainer
│  ├─ testing.py             # Tester
│  ├─ forecasting.py         # Forecaster
│  ├─ io.py                  # 数据加载
│  └─ persistence.py         # 模型保存/加载
├─ run.py                    # CLI 入口（主入口）
├─ main.py                   # 最小示例入口
└─ tests/                    # 重构后测试集
```

## 已接入模型（统一工厂）

`naive`, `arima`, `auto_arima`, `sarima`, `ets`, `theta`, `var`, `bayesian_var`, `linear_var`, `arch`, `garch`, `tbats`, `prophet`, `neuralprophet`, `bayesian_tmt`, `rar`

说明：对可选依赖模型，框架实现了鲁棒回退策略（趋势/Naive），保证管线可运行。

## 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

pytest -q
python run.py --model-name arima --do-train true --do-test true --do-forecast true
```

## CLI 示例

```bash
python run.py \
  --config-module ts_forecast_framework.config.default \
  --config-class AppConfig \
  --model-name var \
  --forecast-horizon 14 \
  --do-train true --do-test true --do-forecast true
```

## 输出结果

- 训练模型：`artifacts/checkpoints/model.pkl`
- 回测结果：`artifacts/test_results/backtest_metrics.csv`
- 预测结果：`artifacts/pred_results/prediction.csv`
- 运行摘要：`artifacts/pred_results/run_summary.json`

## 仓库变更记录

- 2026-03-15：完成统计框架一次性重构（架构、入口、模型工厂、测试全部切换）。
- 2026-03-15：`models/models_ad` 已删除，异常检测资产不在当前主线范围。
