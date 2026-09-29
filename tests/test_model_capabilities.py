import pandas as pd
import pytest

from app.forecasting import Forecaster
from models.factory import ModelFactory
from models.inference import run_point_inference


def test_unsupported_future_input_is_not_silently_ignored():
    with pytest.raises(ValueError, match="future exogenous"):
        Forecaster("naive").forecast(pd.Series([1., 2., 3.], name="y"), 2,
                                     X_future=pd.DataFrame({"temp": [10., 20.]}))


def test_native_dispatch_honors_model_capability(monkeypatch):
    from models.registry import MODEL_REGISTRY
    monkeypatch.setattr(MODEL_REGISTRY["naive"], "supports_native_multistep", False)
    with pytest.raises(ValueError, match="native"):
        run_point_inference(lambda: ModelFactory().create_model("naive"), pd.Series([1., 2.]), 2, "native")


def test_unsupported_native_intervals_fail_explicitly():
    with pytest.raises(ValueError, match="interval"):
        Forecaster("naive", forecast_strategy="native").forecast_with_intervals(pd.Series([1., 2., 3.]), 2)
