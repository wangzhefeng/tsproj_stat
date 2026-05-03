import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory
from models.model.extended_models import NeuralProphetModel, ProphetModel, TBATSModel
from models.model.multivariate import VARModel


def test_seasonal_naive_repeats_last_season():
    y = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    model = ModelFactory().create_model("seasonal_naive", {"season_length": 3})
    model.fit(y)
    pred = model.predict(5)

    assert pred.tolist() == [10.0, 20.0, 30.0, 10.0, 20.0]


def test_historic_average_returns_global_mean():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    model = ModelFactory().create_model("historic_average")
    model.fit(y)
    pred = model.predict(3)

    assert pred.tolist() == [2.5, 2.5, 2.5]


def test_croston_smoke_with_sparse_nonnegative_series():
    y = pd.Series([0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    model = ModelFactory().create_model("croston", {"alpha": 0.2})
    model.fit(y)
    pred = model.predict(4)

    assert len(pred) == 4
    assert (pred >= 0).all()


def test_statsforecast_wrappers_smoke():
    y = pd.Series(np.linspace(1.0, 10.0, 20))
    for name, params in [
        ("dynamic_theta", {"season_length": 1}),
        ("auto_ets", {"season_length": 1}),
        ("auto_theta", {"season_length": 1}),
    ]:
        model = ModelFactory().create_model(name, params)
        model.fit(y)
        pred = model.predict(3)
        assert len(pred) == 3


def test_prophet_model_requires_future_regressors_when_used():
    y = pd.Series(np.linspace(1.0, 12.0, 12), index=pd.date_range("2024-01-01", periods=12, freq="D"), name="y")
    x_hist = pd.DataFrame({"y": y.values, "temperature": np.linspace(10.0, 21.0, 12)})
    model = ProphetModel(regressors=["temperature"])
    model.fit(y, X_hist=x_hist)

    with pytest.raises(ValueError, match="future regressors"):
        model.predict(3)


def test_prophet_model_uses_regressors_with_fake_backend(monkeypatch):
    captured: dict[str, object] = {}

    class FakeProphet:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self.regressors = []

        def add_country_holidays(self, country_name):
            captured["country_name"] = country_name

        def add_regressor(self, name):
            self.regressors.append(name)

        def fit(self, frame):
            captured["fit_columns"] = list(frame.columns)

        def predict(self, future):
            captured["predict_columns"] = list(future.columns)
            return pd.DataFrame({"yhat": [1.0] * len(future)})

    monkeypatch.setattr(ProphetModel, "_import_prophet", staticmethod(lambda: FakeProphet))
    y = pd.Series(np.linspace(1.0, 8.0, 8), index=pd.date_range("2024-01-01", periods=8, freq="D"), name="y")
    x_hist = pd.DataFrame({"y": y.values, "temperature": np.linspace(10.0, 17.0, 8)})
    x_future = pd.DataFrame({"temperature": [18.0, 19.0, 20.0]})
    model = ProphetModel(regressors=["temperature"], country_holidays="US", weekly_seasonality=True)
    model.fit(y, X_hist=x_hist)
    pred = model.predict(3, X_future=x_future)

    assert len(pred) == 3
    assert "temperature" in captured["fit_columns"]
    assert "temperature" in captured["predict_columns"]
    assert captured["country_name"] == "US"


def test_tbats_model_passes_parameters(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTBATSBackend:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit(self, values):
            class Result:
                def forecast(self, steps):
                    return [1.0] * steps

            return Result()

    monkeypatch.setattr(TBATSModel, "_import_tbats", staticmethod(lambda: FakeTBATSBackend))
    model = TBATSModel(
        seasonal_periods=[7, 30],
        use_box_cox=True,
        use_trend=True,
        use_damped_trend=False,
    )
    model.fit(pd.Series(np.linspace(1.0, 20.0, 20)))
    pred = model.predict(2)

    assert len(pred) == 2
    assert captured["kwargs"]["seasonal_periods"] == [7, 30]
    assert captured["kwargs"]["use_box_cox"] is True


def test_neuralprophet_model_fallbacks_cleanly_on_import_failure():
    y = pd.Series(np.linspace(1.0, 10.0, 10))
    model = NeuralProphetModel()
    model.fit(y)
    pred = model.predict(3)

    assert len(pred) == 3


def test_neuralprophet_model_uses_fake_backend(monkeypatch):
    captured: dict[str, object] = {}

    class FakeNeuralProphet:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def add_future_regressor(self, name):
            captured.setdefault("regressors", []).append(name)

        def fit(self, frame, freq=None, minimal=None):
            captured["fit_columns"] = list(frame.columns)
            captured["fit_freq"] = freq
            return pd.DataFrame({"loss": [0.1]})

        def make_future_dataframe(self, df, periods, regressors_df=None):
            captured["future_columns"] = list(regressors_df.columns)
            future = pd.DataFrame({"ds": pd.date_range("2024-01-11", periods=periods, freq="D")})
            for col in regressors_df.columns:
                future[col] = regressors_df[col].values
            return future

        def predict(self, future):
            return future.assign(yhat1=[2.0] * len(future))

    monkeypatch.setattr(NeuralProphetModel, "_import_neuralprophet", staticmethod(lambda: FakeNeuralProphet))
    y = pd.Series(np.linspace(1.0, 10.0, 10), index=pd.date_range("2024-01-01", periods=10, freq="D"), name="y")
    x_hist = pd.DataFrame({"y": y.values, "promo": np.linspace(0.0, 1.0, 10)})
    x_future = pd.DataFrame({"promo": [0.2, 0.4, 0.6]})
    model = NeuralProphetModel(regressors=["promo"])
    model.fit(y, X_hist=x_hist)
    pred = model.predict(3, X_future=x_future)

    assert len(pred) == 3
    assert "promo" in captured["fit_columns"]
    assert "promo" in captured["future_columns"]


def test_var_model_passes_ic_and_maxlags(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResult:
        k_ar = 2

        def forecast(self, input_values, steps):
            return np.asarray([[5.0, 6.0]] * steps)

    class DummyVAR:
        def __init__(self, frame):
            captured["columns"] = list(frame.columns)

        def fit(self, maxlags=None, ic=None):
            captured["maxlags"] = maxlags
            captured["ic"] = ic
            return DummyResult()

    import statsmodels.tsa.api as tsa_api

    monkeypatch.setattr(tsa_api, "VAR", DummyVAR)
    model = VARModel(maxlags=5, ic="aic")
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6, 7], "x": [2, 3, 4, 5, 6, 7, 8]})
    model.fit(df)
    pred = model.predict(2)

    assert len(pred) == 2
    assert captured["maxlags"] == 5
    assert captured["ic"] == "aic"
