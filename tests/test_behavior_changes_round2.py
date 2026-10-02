"""行为变更项回归：sf_auto_arima fallback、ETS damped trend、季节推断下沉。"""
import numpy as np
import pandas as pd
import pytest


def test_sf_auto_arima_fit_failure_falls_back_to_arima(monkeypatch):
    """SF 后端 fit 失败须回退 ARIMAModel 并披露 RuntimeWarning，与 auto_arima 同语义。"""
    import statsforecast.models as sf_models
    from models.model import statsforecast_backend

    class _BrokenAutoARIMA:
        def fit(self, *args, **kwargs):
            raise RuntimeError("simulated statsforecast backend failure")

    monkeypatch.setattr(sf_models, "AutoARIMA", _BrokenAutoARIMA, raising=True)

    y = pd.Series(20 + 0.5 * np.arange(60))
    model = statsforecast_backend.StatsForecastAutoARIMAModel(d=1, max_p=1, max_q=1)
    with pytest.warns(RuntimeWarning, match="fallback"):
        model.fit(y)
    assert model._result is None
    assert model._fallback is not None
    preds = model.predict(3)
    assert len(preds) == 3
    assert np.isfinite(preds.to_numpy(dtype=float)).all()
    info = model.runtime_info()
    assert info.using_fallback_prediction


def test_ets_damped_trend_reaches_statsmodels_backend():
    """damped_trend=True 须传入 ExponentialSmoothing 构造；阻尼路径产出去趋预测。"""
    from models.model.exponential_family import ETSModel
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = pd.Series(50 + 0.8 * np.arange(60) + 0.02 * np.arange(60) ** 1.2)
    model = ETSModel(trend="add", damped_trend=True)
    model.fit(y)
    assert model._result is not None
    # 与后端直连数值一致（证明参数确实传入 damped_trend）
    direct = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()
    np.testing.assert_allclose(model.predict(4), direct.forecast(4))


def test_ets_damped_trend_requires_trend_component():
    from models.model.exponential_family import ETSModel

    with pytest.raises(ValueError, match="damped_trend requires a trend"):
        ETSModel(trend=None, damped_trend=True).fit(pd.Series(np.arange(30.0)))


def test_infer_seasonal_period_lives_in_utils_and_matches_contract():
    """下沉后 utils.seasonality 可独立导入且行为不变：周期序列识别 + 非有限值拒绝。"""
    from utils.seasonality import infer_seasonal_period

    t = np.arange(120)
    seasonal = pd.Series(10 + 3 * np.sin(2 * np.pi * t / 7) + 0.01 * t)
    assert infer_seasonal_period(seasonal) == 7
    with pytest.raises(ValueError, match="non-finite"):
        infer_seasonal_period(pd.Series([1.0, np.nan, 3.0, 4.0] * 5))


def test_models_layer_has_no_data_provider_import():
    """models 包不得 import data_provider（依赖方向：data_provider → models/contracts 或互不依赖）。"""
    import ast
    from pathlib import Path

    root = Path("models")
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("data_provider"):
                offenders.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("data_provider"):
                        offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"models 反向依赖 data_provider: {offenders}"
