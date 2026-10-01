# 预处理

`data_provider/target_transforms/transformer.py`（`TargetTransformer`）统一管理目标变换状态和逆变换，与聚合及缺失修复（[data.md](data.md)）分离。

同包 `seasonality.py` 推断周期、`denoising.py` 去噪、`trend.py` 拟合趋势、`decomposition.py` 分解；`scaling.py` 管理目标缩放。旧模块不保留空壳，迁移表见 [data-architecture.md](data-architecture.md)。

## 目标变换能力

| 层 | 方法 | 说明 |
| --- | --- | --- |
| 去噪 | `moving_average / moving_median` | 轻量方法；LOWESS/Kalman/OnOff 不入主线 |
| 去趋势 | `linear / moving_average` | `moving_average` 趋势窗口当前复用 `denoise_window`（已知耦合，见 [limitations.md](limitations.md)） |
| 分解 | `seasonal_decompose / stl / mstl` | 支持 `trend_resid / resid_only`；MSTL 用独立 `seasonal_periods` 列表、仅加法、窗口须长于最大周期两倍 |
| 缩放 | `scale=true` + `standard / minmax` | 去噪、趋势/分解之后缩放；预测先逆缩放再还原分量；不再由 features 管理目标 |

预处理后建模，预测时逆变换回原始尺度。每窗独立 `fit_transform`，不接触窗口外数据；训练期用 `inverse_transform`，未来预测用 `inverse_forecast`。可逆指趋势/季节还原，去噪丢弃的信息不恢复。

训练归档在同一 checkpoints 实验目录保存 `model.pkl` 与 `target_transformer.pkl`，model_meta 标明输出尺度和变换器路径；离线检查需按 enabled 还原，forecast/test 仍即时拟合，不消费归档。拟合值诊断要求原始目标有限，不以插值值伪造残差。

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
