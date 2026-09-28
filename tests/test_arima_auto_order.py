import pandas as pd
import pytest

from models.factory import ModelFactory


def test_auto_arima_smoke():
    y = pd.Series([1.0, 1.1, 1.3, 1.2, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3, 2.2, 2.4])
    model = ModelFactory().create_model("auto_arima")
    model.fit(y)
    pred = model.predict(2)

    assert len(pred) == 2


def test_auto_arima_does_not_fit_fallback_on_success(monkeypatch):
    class DummyAutoARIMA:
        def predict(self, n_periods: int):
            return [1.0] * n_periods

    import pmdarima as pm

    def fake_auto_arima(*args, **kwargs):
        return DummyAutoARIMA()

    monkeypatch.setattr(pm, "auto_arima", fake_auto_arima)
    model = ModelFactory().create_model("auto_arima")
    model.fit(pd.Series([1.0, 1.2, 1.3, 1.5, 1.7, 1.8]))

    pred = model.predict(2)

    assert model._fallback is None
    assert pred.tolist() == [1.0, 1.0]


def test_auto_arima_forwards_model_params(monkeypatch):
    captured: dict[str, object] = {}

    class DummyAutoARIMA:
        def predict(self, n_periods: int):
            return [0.0] * n_periods

    import pmdarima as pm

    def fake_auto_arima(series, **kwargs):
        captured.update(kwargs)
        return DummyAutoARIMA()

    monkeypatch.setattr(pm, "auto_arima", fake_auto_arima)
    model = ModelFactory().create_model(
        "auto_arima",
        {
            "seasonal": False,
            "m": 1,
            "stepwise": True,
            "start_p": 0,
            "start_q": 0,
            "max_p": 2,
            "max_q": 2,
            "max_order": 4,
            "maxiter": 20,
            "information_criterion": "aic",
            "trace": True,
            "error_action": "ignore",
            "suppress_warnings": True,
        },
    )

    model.fit(pd.Series([1.0, 1.1, 1.2, 1.25, 1.3, 1.35]))

    assert captured["seasonal"] is False
    assert captured["m"] == 1
    assert captured["stepwise"] is True
    assert captured["start_p"] == 0
    assert captured["start_q"] == 0
    assert captured["max_p"] == 2
    assert captured["max_q"] == 2
    assert captured["max_order"] == 4
    assert captured["maxiter"] == 20
    assert captured["information_criterion"] == "aic"
    assert captured["trace"] is True
    assert captured["error_action"] == "ignore"
    assert captured["suppress_warnings"] is True


def test_sarima_forwards_model_params(monkeypatch):
    captured_init: dict[str, object] = {}
    captured_fit: dict[str, object] = {}

    class DummyResult:
        def forecast(self, steps: int):
            return [0.0] * steps

    class DummySARIMAX:
        def __init__(self, series, **kwargs):
            captured_init.update(kwargs)

        def fit(self, **kwargs):
            captured_fit.update(kwargs)
            return DummyResult()

    import statsmodels.tsa.statespace.sarimax as sarimax_module

    monkeypatch.setattr(sarimax_module, "SARIMAX", DummySARIMAX)
    model = ModelFactory().create_model(
        "sarima",
        {
            "order": (2, 1, 0),
            "seasonal_order": (0, 1, 1, 7),
            "trend": "c",
            "enforce_stationarity": False,
            "enforce_invertibility": False,
            "simple_differencing": True,
            "fit_kwargs": {"disp": False, "maxiter": 15, "method": "lbfgs"},
        },
    )

    model.fit(pd.Series([1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45]))

    assert captured_init["order"] == (2, 1, 0)
    assert captured_init["seasonal_order"] == (0, 1, 1, 7)
    assert captured_init["trend"] == "c"
    assert captured_init["enforce_stationarity"] is False
    assert captured_init["enforce_invertibility"] is False
    assert captured_init["simple_differencing"] is True
    assert captured_fit["disp"] is False
    assert captured_fit["maxiter"] == 15
    assert captured_fit["method"] == "lbfgs"
