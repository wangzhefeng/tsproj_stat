import numpy as np
import pandas as pd
import pytest

from artifacts.metadata import model_info_payload
from models.model.extended_models import ProphetModel
from models.factory import ModelFactory


class LinearProphetBackend:
    """受控后端仅替代可选依赖，状态判断及预测路由使用真实包装器。"""
    def __init__(self, **kwargs):
        pass

    def fit(self, frame):
        self.last = float(frame.y.iloc[-1])

    def predict(self, future):
        return pd.DataFrame({"yhat": self.last + np.arange(1, len(future) + 1)})


def test_prophet_metadata_matches_native_then_failed_then_recovered_fit(monkeypatch):
    model = ProphetModel()
    y = pd.Series(np.arange(20.))
    monkeypatch.setattr(model, "_import_prophet", lambda: LinearProphetBackend)
    model.fit(y)
    np.testing.assert_allclose(model.predict(2), [20., 21.])
    info = model_info_payload(model, {}, "prophet")
    assert info["using_fallback_prediction"] is False
    assert info["has_native_result"] is True

    def unavailable():
        raise ImportError("controlled backend failure")

    monkeypatch.setattr(model, "_import_prophet", unavailable)
    with pytest.warns(RuntimeWarning):
        model.fit(y)
    info = model_info_payload(model, {}, "prophet")
    assert info["using_fallback_prediction"] is True
    assert "controlled backend failure" in info["fallback_reason"]
    monkeypatch.setattr(model, "_import_prophet", lambda: LinearProphetBackend)
    model.fit(y)
    info = model_info_payload(model, {}, "prophet")
    assert info["using_fallback_prediction"] is False
    assert info["fallback_reason"] is None


@pytest.mark.parametrize("name,params,y,expected", [
    ("naive", {}, [1., 3., 5., 7.], [7., 7.]),
    ("random_walk_drift", {}, [1., 3., 5., 7.], [9., 11.]),
])
def test_baseline_and_sf_metadata_do_not_report_fallback(name, params, y, expected):
    model = ModelFactory().create_model(name, params).fit(pd.Series(y))
    np.testing.assert_allclose(model.predict(2), expected)
    info = model_info_payload(model, params, name)
    assert info["using_fallback_prediction"] is False
    if name == "random_walk_drift":
        assert info["has_native_result"] is True


def test_statsmodels_native_and_trainer_fallback(monkeypatch):
    from pipeline.trainer import Trainer
    from models.model.arima_family import ARIMAModel
    model = ARIMAModel(order=(0, 0, 0)).fit(pd.Series([2., 2., 2., 2., 2., 2.]))
    assert model.runtime_info().has_native_result
    np.testing.assert_allclose(model.predict(2), [2., 2.], atol=1e-3)
    def fail(*args, **kwargs):
        raise RuntimeError("controlled fit failure")
    monkeypatch.setattr(ARIMAModel, "fit", fail)
    fallback = Trainer("arima").train(pd.Series([1., 2., 3.]))
    assert fallback.runtime_info().is_trainer_fallback
    assert fallback.runtime_info().using_fallback_prediction
    assert fallback.runtime_info().fallback_reason == "controlled fit failure"
    np.testing.assert_allclose(fallback.predict(2), [3., 3.])


def test_neuralprophet_runtime_tracks_native_failure_recovery(monkeypatch):
    from models.model.extended_models import NeuralProphetModel
    class Backend(LinearProphetBackend):
        def fit(self, frame, **kwargs):
            super().fit(frame)
        def make_future_dataframe(self, df, periods, **kwargs):
            return pd.DataFrame({"ds": pd.date_range(df.ds.iloc[-1], periods=periods + 1)[1:]})
    model = NeuralProphetModel()
    monkeypatch.setattr(model, "_import_neuralprophet", lambda: Backend)
    model.fit(pd.Series(np.arange(20.)))
    np.testing.assert_allclose(model.predict(2), [20., 21.])
    assert model.runtime_info().has_native_result
    def unavailable():
        raise ImportError("controlled neural backend unavailable")
    monkeypatch.setattr(model, "_import_neuralprophet", unavailable)
    with pytest.warns(RuntimeWarning):
        model.fit(pd.Series(np.arange(20.)))
    assert model.runtime_info().using_fallback_prediction
    monkeypatch.setattr(model, "_import_neuralprophet", lambda: Backend)
    model.fit(pd.Series(np.arange(20.)))
    assert model.runtime_info().fallback_reason is None
    np.testing.assert_allclose(model.predict(2), [20., 21.])
