import numpy as np
import pandas as pd
import pytest

from app.forecasting import Forecaster
from models.selector import AutoSelector


def test_history_covariates_require_capability_or_explicit_ignore():
    y = pd.Series([1., 2., 3.], name="y")
    x = pd.DataFrame({"y": y, "temp": [1., 3., 8.]})
    with pytest.raises(ValueError, match="historical covariates"):
        Forecaster("naive").forecast(y, 2, X_hist=x)
    ignored = Forecaster("naive", ignore_unsupported_inputs=True).forecast(y, 2, X_hist=x)
    np.testing.assert_allclose(ignored, [3., 3.])


def test_selector_filters_capabilities_and_passes_covariates():
    rng = np.random.default_rng(12)
    x = rng.normal(size=60)
    frame = pd.DataFrame({"y": 3 * x, "temp": x})
    selector = AutoSelector(candidates=["naive", "linear_var"], initial_train_size=30,
                            horizon=2, n_windows=2, forecast_strategy="native",
                            model_params_map={"linear_var": {"feature_lags": [0]}})
    result = selector.select(frame.y, X_hist=frame, future_exog_cols=["temp"])
    assert result == "linear_var"
    assert "naive" not in selector.scores
    assert selector.scores["linear_var"] < 1e-8
