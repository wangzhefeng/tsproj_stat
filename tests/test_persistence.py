import pandas as pd
import json
from importlib.metadata import version
import numpy as np

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


def test_statsforecast_archive_records_backend_and_preserves_prediction(tmp_path):
    model = ModelFactory().create_model("random_walk_drift").fit(pd.Series([1., 3., 5., 7.]))
    path = tmp_path / "model.pkl"
    save_model(model, str(path))
    meta = json.loads((tmp_path / "model_meta.json").read_text())
    assert meta["key_deps"]["statsforecast"] == version("statsforecast")
    np.testing.assert_allclose(load_model(str(path)).predict(3), [9., 11., 13.])
