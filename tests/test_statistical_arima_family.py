import warnings

import pandas as pd
import pytest

from models.factory import ModelFactory
from models.model import ARIMAModel, AutoARIMAModel


def test_arima_model_suppresses_known_initialization_warnings():
    series = pd.Series([1.0, 1.1, 1.3, 1.2, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3, 2.2, 2.4])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(series)
        pred = model.predict(2)

    assert len(pred) == 2
    messages = [str(item.message) for item in caught]
    assert not any("Non-invertible starting MA parameters found" in msg for msg in messages)
    assert not any("Non-stationary starting autoregressive parameters found" in msg for msg in messages)


def test_auto_arima_model_fallback_still_predicts():
    series = pd.Series([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.2])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = AutoARIMAModel(seasonal=True, m=12)
        model.fit(series)
        pred = model.predict(2)

    assert len(pred) == 2
    messages = [str(item.message) for item in caught]
    assert not any("Non-invertible starting MA parameters found" in msg for msg in messages)
    assert not any("Non-stationary starting autoregressive parameters found" in msg for msg in messages)


@pytest.mark.parametrize(
    ("model_name", "params", "expected_order"),
    [
        ("ar", {"p": 2}, (2, 0, 0)),
        ("ma", {"q": 2}, (0, 0, 2)),
        ("arma", {"p": 2, "q": 1}, (2, 0, 1)),
    ],
)
def test_specialized_arima_family_models_map_to_expected_orders(monkeypatch, model_name, params, expected_order):
    captured: dict[str, object] = {}

    class DummyResult:
        def forecast(self, steps: int):
            return [0.0] * steps

    class DummyARIMA:
        def __init__(self, series, order):
            captured["order"] = order

        def fit(self):
            return DummyResult()

    import statsmodels.tsa.arima.model as arima_module

    monkeypatch.setattr(arima_module, "ARIMA", DummyARIMA)
    model = ModelFactory().create_model(model_name, params)
    model.fit(pd.Series([1.0, 1.2, 1.1, 1.3, 1.6, 1.8]))

    assert captured["order"] == expected_order
