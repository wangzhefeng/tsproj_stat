from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pandas as pd
import pytest

from artifacts.identity import ArtifactIdentity, build_identity, check_field_classification, frame_fingerprint
from artifacts.paths import build_experiment_path, plan_run_artifacts, prepare_run_artifacts
from config import AppConfig
from config.model_params import resolve_model_params


def test_classification_and_equivalent_configs(tmp_path):
    check_field_classification()
    cfg = AppConfig(results_dir=str(tmp_path), model_name="ets", model_params={"alpha": 1, "beta": 2})
    equivalent = replace(cfg, model_params={"beta": 2, "alpha": 1}, backtest_train_size=cfg.history_size,
                         scaler_type="minmax", interval_alpha=0.2, simulate_n_paths=300)
    assert build_experiment_path(cfg) == build_experiment_path(equivalent)
    assert resolve_model_params(replace(cfg, model_params={"validation_size": 5}, ets_validation_size=10))["validation_size"] == 5
    plan_run_artifacts(cfg, "planning-only")
    assert not list(tmp_path.iterdir())


def test_strict_identity_and_distinct_source():
    with pytest.raises(ValueError, match="non-finite"):
        build_identity(AppConfig(model_params={"x": float("nan")}))
    assert build_identity(AppConfig(data_path="one/data.csv")) != build_identity(AppConfig(data_path="two/data.csv"))
    first = pd.DataFrame({"y": [1., 2.]})
    assert frame_fingerprint(first) != frame_fingerprint(first.assign(y=[1., 3.]))


def test_short_identity_collision_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ArtifactIdentity, "token", property(lambda self: "identity-v2-collision"))
    cfg = AppConfig(results_dir=str(tmp_path))
    prepare_run_artifacts(cfg)
    with pytest.raises(ValueError, match="collision"):
        prepare_run_artifacts(replace(cfg, target_col="other"))


def test_identity_concurrent_creation(tmp_path):
    cfg = AppConfig(results_dir=str(tmp_path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(lambda _: prepare_run_artifacts(cfg), range(24)))
    assert len({item.run_id for item in runs}) == 24


def test_tuple_parameters_are_reusable(tmp_path):
    cfg = AppConfig(results_dir=str(tmp_path), model_params={"order": (1, 1, 0)})
    prepare_run_artifacts(cfg)
    prepare_run_artifacts(cfg)
