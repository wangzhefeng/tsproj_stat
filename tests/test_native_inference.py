import numpy as np
import pandas as pd
import pytest

from forecasting.strategies import run_point_inference, run_interval_inference
from models.model.baseline_models import AutoETSModel


@pytest.mark.parametrize("strategy,expected_fits", [("native", 1), ("direct", 4), ("single_step", 1)])
@pytest.mark.parametrize("intervals", [False, True])
def test_prediction_uses_only_required_fits(monkeypatch, strategy, expected_fits, intervals):
    """保护一次原点执行的计算预算，同时对照原生后端的数值契约。"""
    from statsforecast.models import AutoETS

    y = pd.Series(20 + np.arange(60) * 0.2 + np.sin(np.arange(60)))
    h = 1 if strategy == "single_step" else 4
    expected = AutoETS(season_length=1).fit(y.to_numpy()).predict(h, level=[95])
    calls = []
    original = AutoETSModel.fit

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AutoETSModel, "fit", counted)
    if intervals:
        pred = run_interval_inference(AutoETSModel, y, h, strategy)
        np.testing.assert_allclose(pred["yhat_lower"], expected["lo-95"])
        np.testing.assert_allclose(pred["yhat_upper"], expected["hi-95"])
        actual = pred["yhat"]
    else:
        actual = run_point_inference(AutoETSModel, y, h, strategy)
    np.testing.assert_allclose(actual, expected["mean"])
    assert len(calls) == expected_fits


@pytest.mark.parametrize("alpha", [0, 1, float("nan")])
def test_intervals_reject_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        run_interval_inference(AutoETSModel, pd.Series(range(30)), 2, "direct", alpha=alpha)


def test_native_does_not_accept_missing_backend_intervals():
    from models.factory import ModelFactory
    from models.model.arima_family import ARIMAModel
    from unittest.mock import patch
    with patch.object(ARIMAModel, "predict_with_intervals", return_value=pd.DataFrame({
        "yhat": [1., 1.], "yhat_lower": [np.nan, np.nan], "yhat_upper": [np.nan, np.nan]})):
        with pytest.raises(ValueError, match="interval"):
            run_interval_inference(lambda: ModelFactory().create_model("arima", {"order": [0, 0, 0]}),
                                   pd.Series(range(30)), 2, "native")
