import json
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from models.factory import ModelFactory
from artifacts.checkpoints import load_model, save_model


def test_archive_checksum_rejects_corruption(tmp_path):
    path = tmp_path / "model.pkl"
    model = ModelFactory().create_model("naive").fit(pd.Series([1., 2., 3.]))
    save_model(model, str(path))
    assert (tmp_path / "checkpoint_manifest.json").is_file()
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_model(str(path))


def test_failed_archive_is_not_legacy(tmp_path, monkeypatch):
    from artifacts import checkpoints
    path = tmp_path / "model.pkl"
    def fail(*args, **kwargs):
        raise OSError("controlled failure")
    monkeypatch.setattr(checkpoints, "write_pickle", fail)
    with pytest.raises(OSError):
        save_model(object(), str(path))
    with pytest.raises(ValueError, match="incomplete"):
        load_model(str(path))


def test_moved_bundle_and_legacy(tmp_path):
    from artifacts.checkpoints import save_checkpoint, load_checkpoint
    from data_provider.target_transforms.transformer import TargetTransformer
    y = pd.Series([1., 3., 5., 7., 9., 11.])
    processor = TargetTransformer(detrend_method="linear")
    transformed = processor.fit_transform(y)
    model = ModelFactory().create_model("naive").fit(transformed)
    original = tmp_path / "original"
    save_checkpoint(model, processor, original, {})
    moved = tmp_path / "moved"
    shutil.move(str(original), moved)
    loaded, transformer, meta = load_checkpoint(moved)
    np.testing.assert_allclose(transformer.inverse_forecast(loaded.predict(2)), [13., 15.], atol=1e-8)
    assert meta["target_transformer_path"] == "target_transformer.pkl"
    legacy = tmp_path / "legacy" / "model.pkl"
    legacy.parent.mkdir()
    legacy.write_bytes(pickle.dumps(model))
    np.testing.assert_allclose(load_model(str(legacy)).predict(2), model.predict(2))


@pytest.mark.parametrize("name,old_module", [
    (name, "models.model.baseline_models") for name in
    ["AutoETSModel", "AutoCESModel", "AutoThetaModel", "DynamicThetaModel", "RandomWalkWithDriftModel", "SeasonalWindowAverageModel"]
] + [("StatsForecastAutoARIMAModel", "models.model.arima_family")])
def test_legacy_statsforecast_class_paths_load(tmp_path, name, old_module):
    from models.model import statsforecast_backend
    cls = getattr(statsforecast_backend, name)
    # protocol 0 GLOBAL 的 module/name 与历史 pickle 的解析路径相同；不加载不可信字节。
    payload = pickle.dumps(cls, protocol=0).replace(b"models.model.statsforecast_backend", old_module.encode())
    path = tmp_path / "legacy.pkl"
    path.write_bytes(payload)
    assert load_model(str(path)) is cls


def test_legacy_statsforecast_state_predicts_identically(tmp_path):
    from models.model.statsforecast_backend import AutoETSModel
    model = AutoETSModel(season_length=1, model="ANN").fit(pd.Series(10 + np.sin(np.arange(40.))))
    payload = pickle.dumps(model, protocol=0).replace(b"models.model.statsforecast_backend", b"models.model.baseline_models")
    path = tmp_path / "legacy.pkl"
    path.write_bytes(payload)
    np.testing.assert_allclose(load_model(str(path)).predict(3), model.predict(3))
