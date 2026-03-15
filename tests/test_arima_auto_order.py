import pandas as pd

from task.models.factory import ModelFactory


def test_auto_arima_smoke():
    y = pd.Series([1.0, 1.1, 1.3, 1.2, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3, 2.2, 2.4])
    model = ModelFactory().create_model("auto_arima")
    model.fit(y)
    pred = model.predict(2)

    assert len(pred) == 2
