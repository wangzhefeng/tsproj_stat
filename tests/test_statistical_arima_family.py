import warnings

import pandas as pd

from models.model import ARIMAModel, AutoARIMAModel


def test_arima_model_suppresses_known_initialization_warnings():
    series = pd.Series([1.0, 1.1, 1.3, 1.2, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3, 2.2, 2.4])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(series)
        pred = model.predict(2)

    assert len(pred) == 2
    messages = [str(item.message) for item in caught]
    assert not any("Non-invertible starting MA parameters found" in msg for msg in messages)
    assert not any("Non-stationary starting autoregressive parameters found" in msg for msg in messages)


def test_auto_arima_model_fallback_still_predicts():
    series = pd.Series([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.2])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = AutoARIMAModel(seasonal=True, m=12)
        model.fit(series)
        pred = model.predict(2)

    assert len(pred) == 2
    messages = [str(item.message) for item in caught]
    assert not any("Non-invertible starting MA parameters found" in msg for msg in messages)
    assert not any("Non-stationary starting autoregressive parameters found" in msg for msg in messages)
