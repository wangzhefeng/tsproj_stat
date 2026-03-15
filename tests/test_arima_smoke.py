import pandas as pd

from models.factory import ModelFactory


def test_arima_smoke():
    y = pd.Series([1.0, 1.2, 1.1, 1.3, 1.6, 1.8, 2.0, 2.2, 2.1, 2.4])
    model = ModelFactory().create_model("arima", {"order": (1, 1, 0)})
    model.fit(y)
    pred = model.predict(3)

    assert len(pred) == 3
