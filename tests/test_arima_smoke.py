import pandas as pd

from ts_forecast_framework.models.statistical import ARIMAForecaster


def test_arima_smoke_predict_length():
    y = pd.Series([1.0, 1.2, 1.1, 1.4, 1.6, 1.7, 1.9, 2.0, 2.2, 2.3])
    model = ARIMAForecaster(order=(1, 1, 1)).fit(y)
    pred = model.predict(3)
    assert len(pred) == 3
