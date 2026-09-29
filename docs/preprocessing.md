# 预处理

可逆预处理统一放在 `data_provider/data_processor.py`（`DataProcessor`），与聚合（[data.md](data.md#频率聚合)）严格分离。

## 三层能力

| 层 | 方法 | 说明 |
| --- | --- | --- |
| 去噪 | `moving_average / moving_median` | 轻量方法；LOWESS/Kalman/OnOff 不入主线 |
| 去趋势 | `linear / moving_average` | `moving_average` 趋势窗口当前复用 `denoise_window`（已知耦合，见 [limitations.md](limitations.md)） |
| 分解 | `seasonal_decompose / stl / mstl` | 支持 `trend_resid / resid_only`；MSTL 用独立 `seasonal_periods` 列表、仅加法、窗口须长于最大周期两倍 |

预处理后建模，预测时逆变换回原始尺度。每窗独立 `fit_transform`，不接触窗口外数据。

## 示例

```bash
.venv/bin/python run.py --denoise_method moving_median --denoise_window 5 --detrend_method linear
```

ARIMA 家族分解预处理（对残差建模后重组趋势/季节）：

```bash
.venv/bin/python run.py \
  --model_name ar --model_params '{"p":2}' \
  --decomposition_method seasonal_decompose --decomposition_target resid_only \
  --predict_horizon 4
```

预处理后 EDA 对比（`--eda_run_preprocessed true`）：

```bash
.venv/bin/python run.py --do_eda true --do_train false --do_test false --do_forecast false \
  --decomposition_method seasonal_decompose --decomposition_target resid_only \
  --seasonal_period 7 --eda_run_preprocessed true
```

MSTL 多季节示例见 [strategies.md](strategies.md#原生多步区间与多季节分解)。

## ETS 季节周期来源

`ETSModel`（SES/DES/TES 统一入口）的 `seasonal_periods` 优先级：`model_params.seasonal_periods` > `--seasonal_period` > 自动周期推断；启用了 `seasonal` 但推断失败直接报错，不静默退化。

ETS 调参示例（smoothing grid）：

```bash
.venv/bin/python run.py --model_name ets \
  --model_params '{"trend":"add","seasonal":"add"}' --seasonal_period 5 \
  --ets_tune_smoothing_params true \
  --ets_smoothing_grid_level 0.2,0.5,0.8 --ets_smoothing_grid_trend 0.2,0.5 \
  --ets_smoothing_grid_seasonal 0.2,0.5 \
  --predict_horizon 4 --do_train false --do_test false --do_forecast true
```
