import pandas as pd

from ts_forecast_framework.models.statistical import ARIMAForecaster


def test_arima_auto_order_smoke():
    y = pd.Series([1.0, 1.1, 1.3, 1.2, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3, 2.2, 2.4])
    model = ARIMAForecaster(
        auto_order=True,
        order_grid=[(0, 1, 0), (1, 1, 0), (1, 1, 1)],
        ic="aic",
    ).fit(y)

    assert model.selected_order in {(0, 1, 0), (1, 1, 0), (1, 1, 1)}
    assert len(model.predict(2)) == 2
