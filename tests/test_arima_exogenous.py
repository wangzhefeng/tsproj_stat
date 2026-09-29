import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(5)
    x = pd.DataFrame({"temp": rng.normal(size=80), "calendar": rng.normal(size=80)})
    y = pd.Series(3 * x.temp - 2 * x.calendar + rng.normal(scale=0.01, size=80), name="y")
    return y, x


@pytest.mark.parametrize("name,params", [
    ("arima", {"order": [0, 0, 0]}),
    ("sarima", {"order": [0, 0, 0], "seasonal_order": [0, 0, 0, 7]}),
    ("auto_arima", {"d": 0, "start_p": 0, "start_q": 0, "max_p": 0, "max_q": 0}),
    ("sf_auto_arima", {"d": 0, "max_p": 0, "max_q": 0}),
])
def test_arima_future_regression_and_schema(name, params, regression_data):
    y, x = regression_data
    model = ModelFactory().create_model(name, params).fit(y, X_hist=x)
    future = pd.DataFrame({"calendar": [0., 1., 2.], "temp": [2., 4., 6.]})
    actual = model.predict(3, X_future=future)
    np.testing.assert_allclose(actual, [6., 10., 14.], atol=0.1)
    intervals = model.predict_with_intervals(3, X_future=future)
    np.testing.assert_allclose(intervals.yhat, actual)
    assert np.isfinite(intervals.to_numpy()).all()
    with pytest.raises(ValueError, match="exogenous"):
        model.predict(3)
    with pytest.raises(ValueError, match="exogenous"):
        model.predict(3, X_future=future.iloc[:2])
    with pytest.raises(ValueError, match="exogenous"):
        model.predict(3, X_future=future.drop(columns="temp"))


@pytest.mark.parametrize("target_name", [None, "load"])
def test_recursive_preserves_custom_target_identity(regression_data, target_name):
    from app.forecasting import Forecaster
    y, x = regression_data
    y = y.rename(target_name)
    future = pd.DataFrame({"calendar": [0., 1., 2.], "temp": [2., 4., 6.]})
    pred = Forecaster("arima", {"order": [0, 0, 0]}, "recursive").forecast(y, 3, X_hist=x, X_future=future)
    np.testing.assert_allclose(pred, [6., 10., 14.], atol=0.1)


def test_order_search_uses_same_exogenous_regression(regression_data):
    from statsmodels.tsa.arima.model import ARIMA
    y, x = regression_data
    expected = ARIMA(y, exog=x.to_numpy(), order=(0, 0, 0)).fit().aic
    model = ModelFactory().create_model("arima", {"auto_order": True, "order_grid": [[0, 0, 0]]}).fit(y, X_hist=x)
    assert model.selected_score == pytest.approx(expected)


def test_training_does_not_hide_invalid_exogenous_input(regression_data):
    from app.training import Trainer
    y, x = regression_data
    with pytest.raises(ValueError, match="exogenous"):
        Trainer("arima").train(y, X_hist=x, X_future=pd.DataFrame({"wrong": [1.]}))


@pytest.mark.parametrize("name", ["arima", "sarima", "auto_arima"])
@pytest.mark.parametrize("intervals", [False, True])
def test_fallback_still_enforces_future_schema(regression_data, monkeypatch, name, intervals):
    y, x = regression_data
    if name == "auto_arima":
        import pmdarima
        owner, attr = pmdarima, "auto_arima"
    elif name == "sarima":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        owner, attr = SARIMAX, "fit"
    else:
        from statsmodels.tsa.arima.model import ARIMA
        owner, attr = ARIMA, "fit"

    def fail(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(owner, attr, fail)
    model = ModelFactory().create_model(name)
    with pytest.warns(RuntimeWarning, match="fallback"):
        model.fit(y, X_hist=x)
    predict = model.predict_with_intervals if intervals else model.predict
    with pytest.raises(ValueError, match="exogenous"):
        predict(2, X_future=pd.DataFrame({"wrong": [1., 2.]}))


def test_auto_arima_fallback_keeps_regression_inputs(regression_data, monkeypatch):
    import pmdarima
    y, x = regression_data

    def fail(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(pmdarima, "auto_arima", fail)
    model = ModelFactory().create_model("auto_arima")
    with pytest.warns(RuntimeWarning, match="fallback"):
        model.fit(y, X_hist=x)
    future = pd.DataFrame({"temp": [2., 4.], "calendar": [0., 1.]})
    np.testing.assert_allclose(model.predict(2, X_future=future), [6., 10.], atol=0.1)
