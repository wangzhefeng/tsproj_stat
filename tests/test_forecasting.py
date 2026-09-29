"""forecast 输出校验的显式策略测试（P14 / T14）。"""

import numpy as np
import pandas as pd
import pytest

from forecasting.forecaster import _validate_forecast


def test_validate_forecast_raises_on_nan_by_default():
    yhat = pd.Series([1.0, np.nan, 3.0])

    with pytest.raises(ValueError, match="NaN in forecast output"):
        _validate_forecast(yhat, horizon=3, model_name="dummy")


def test_validate_forecast_fills_nan_only_when_explicitly_allowed():
    yhat = pd.Series([1.0, np.nan, 3.0])

    filled = _validate_forecast(yhat, horizon=3, model_name="dummy", allow_nan_fill=True)

    assert filled.tolist() == [1.0, 1.0, 3.0]


def test_validate_forecast_raises_on_inf_regardless_of_allowance():
    yhat = pd.Series([1.0, np.inf, 3.0])

    with pytest.raises(ValueError, match="inf values"):
        _validate_forecast(yhat, horizon=3, model_name="dummy", allow_nan_fill=True)


def test_validate_forecast_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="forecast length"):
        _validate_forecast(pd.Series([1.0, 2.0]), horizon=3, model_name="dummy")
