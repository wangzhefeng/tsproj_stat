import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory


@pytest.mark.parametrize("name,params,y,expected", [
    ("random_walk_drift", {}, [1., 3., 5., 7.], [9., 11., 13.]),
    ("seasonal_window_average", {"season_length": 3, "window_size": 2}, [1., 2., 3., 5., 6., 7.], [3., 4., 5.]),
])
def test_new_baselines_have_analytic_forecasts(name, params, y, expected):
    model = ModelFactory().create_model(name, params).fit(pd.Series(y))
    np.testing.assert_allclose(model.predict(3), expected)


def test_auto_ces_adapter_matches_upstream():
    from statsforecast.models import AutoCES
    y = 30 + np.arange(60) * 0.1 + np.sin(np.arange(60))
    expected = AutoCES(season_length=1).fit(y).predict(4)["mean"]
    model = ModelFactory().create_model("auto_ces").fit(pd.Series(y))
    np.testing.assert_allclose(model.predict(4), expected)
