import pandas as pd

from ts_forecast_framework.models.statistical import NaiveForecaster
from ts_forecast_framework.persistence import load_model, save_model


def test_save_and_load_model(tmp_path):
    model = NaiveForecaster().fit(pd.Series([1, 2, 3]))
    model_path = tmp_path / "naive.pkl"

    save_model(model, str(model_path))
    loaded = load_model(str(model_path))

    pred = loaded.predict(3)
    assert len(pred) == 3
    assert float(pred.iloc[0]) == 3.0
