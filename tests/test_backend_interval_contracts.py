"""统计后端适配必须保留其点预测、区间及非整数置信水平。"""
import numpy as np
import pandas as pd
import pytest

from models.model.exponential_family import ThetaModel
from models.model.multivariate import VARModel
from models.model.arima_family import StatsForecastAutoARIMAModel


def test_theta_intervals_match_statsmodels():
    from statsmodels.tsa.forecasting.theta import ThetaModel as Backend

    y = pd.Series(20 + np.arange(60) * 0.1 + np.sin(np.arange(60)))
    expected = Backend(y, period=1).fit()
    model = ThetaModel(period=1).fit(y)
    actual = model.predict_with_intervals(4, alpha=0.125)
    np.testing.assert_allclose(actual.yhat, expected.forecast(4))
    np.testing.assert_allclose(actual[['yhat_lower', 'yhat_upper']], expected.prediction_intervals(4, alpha=0.125))


def test_var_intervals_preserve_backend_order():
    from statsmodels.tsa.api import VAR

    frame = pd.DataFrame(np.random.default_rng(7).normal(size=(100, 2)), columns=['y', 'x'])
    expected = VAR(frame).fit(maxlags=1)
    mean, lower, upper = expected.forecast_interval(frame.to_numpy()[-1:], 4, alpha=0.125)
    model = VARModel(maxlags=1).fit(frame.y, X_hist=frame)
    actual = model.predict_with_intervals(4, alpha=0.125)
    np.testing.assert_allclose(actual.yhat, mean[:, 0])
    np.testing.assert_allclose(actual.yhat_lower, lower[:, 0])
    np.testing.assert_allclose(actual.yhat_upper, upper[:, 0])


@pytest.mark.parametrize('alpha', [0.05, 0.125])
def test_sf_auto_arima_preserves_fractional_level(alpha):
    from statsforecast.models import AutoARIMA

    y = pd.Series(20 + np.arange(60) * 0.1 + np.sin(np.arange(60)))
    model = StatsForecastAutoARIMAModel(d=1, max_p=1, max_q=1).fit(y)
    backend = AutoARIMA(d=1, seasonal=False, max_p=1, max_q=1, start_p=1, start_q=1).fit(y.to_numpy())
    level = round(100 * (1 - alpha), 10)
    expected = backend.predict(4, level=[level])
    actual = model.predict_with_intervals(4, alpha=alpha)
    np.testing.assert_allclose(actual.yhat, expected['mean'])
    np.testing.assert_allclose(actual.yhat_lower, expected[f'lo-{level}'])
    np.testing.assert_allclose(actual.yhat_upper, expected[f'hi-{level}'])
