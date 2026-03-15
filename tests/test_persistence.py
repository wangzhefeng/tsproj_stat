import pandas as pd

from models.factory import ModelFactory
from models.persistence import load_model, save_model


def test_save_and_load_model(tmp_path):
    model = ModelFactory().create_model("naive")
    model.fit(pd.Series([1, 2, 3]))
    model_path = tmp_path / "naive.pkl"

    save_model(model, str(model_path))
    loaded = load_model(str(model_path))

    pred = loaded.predict(3)
    assert len(pred) == 3
    assert float(pred.iloc[0]) == 3.0
