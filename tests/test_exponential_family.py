import pandas as pd
import pytest

from models.factory import ModelFactory
from models.model.exponential_family import ETSModel


def test_ets_model_smoke_for_ses_des_tes():
    y = pd.Series(
        [10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 14.0, 13.5, 14.5, 15.0, 16.0, 15.5] * 2
    )

    ses = ModelFactory().create_model("ets", {"trend": None, "seasonal": None})
    des = ModelFactory().create_model("ets", {"trend": "add", "seasonal": None})
    tes = ModelFactory().create_model(
        "ets",
        {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
    )

    for model in [ses, des, tes]:
        model.fit(y)
        pred = model.predict(3)
        assert len(pred) == 3


def test_ets_model_tuning_uses_best_grid_candidate(monkeypatch):
    seen: list[tuple[float | None, float | None, float | None]] = []

    class DummyResult:
        def __init__(self, level: float | None):
            self.level = level

        def forecast(self, steps: int):
            base = self.level if self.level is not None else 0.0
            return [base] * steps

    class DummyETS:
        def __init__(self, series, trend=None, seasonal=None, seasonal_periods=None):
            self.series = series

        def fit(
            self,
            smoothing_level=None,
            smoothing_trend=None,
            smoothing_seasonal=None,
            optimized=True,
        ):
            seen.append((smoothing_level, smoothing_trend, smoothing_seasonal))
            return DummyResult(smoothing_level)

    import statsmodels.tsa.holtwinters as hw_module

    monkeypatch.setattr(hw_module, "ExponentialSmoothing", DummyETS)
    model = ETSModel(
        trend=None,
        seasonal=None,
        tune_smoothing_params=True,
        smoothing_grid_level=[0.2, 0.8],
        validation_size=3,
    )

    model.fit(pd.Series([1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8]))
    pred = model.predict(2)

    assert len(pred) == 2
    assert seen[-1] == (0.8, None, None)
    assert pred.tolist() == [0.8, 0.8]


def test_ets_model_requires_seasonal_period_when_seasonal_enabled():
    with pytest.raises(ValueError, match="seasonal_periods"):
        ETSModel(trend="add", seasonal="add").fit(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
