from dataclasses import replace

import pytest

from config import AppConfig
from artifacts.paths import build_experiment_path, build_eda_path, prepare_run_artifacts, resolve_data_name


@pytest.mark.parametrize("changes", [
    {"interval_levels": [0.8]}, {"target_col": "load"}, {"freq": "h"},
    {"decomposition_target": "resid_only"}, {"forecast_use_update": True},
    {"simulate_enabled": True}, {"future_exog_path": "future-other.csv"},
])
def test_distinct_effective_configs_have_distinct_paths(changes):
    base = AppConfig(return_intervals=True, decomposition_method="stl", seasonal_period=7)
    assert build_experiment_path(base) != build_experiment_path(replace(base, **changes))


def test_eda_preprocessing_and_comparison_identity():
    base = AppConfig(eda_run_preprocessed=True)
    assert build_eda_path(base) != build_eda_path(replace(base, detrend_method="linear"))
    assert build_eda_path(base) != build_eda_path(replace(base, eda_comparison_paths=["other.csv"]))
    assert build_eda_path(base) == build_eda_path(replace(base, model_name="naive"))


@pytest.mark.parametrize("name", [" ", ".", "../x", "/x", "~x", "a\\b", "a/../b", "x\x00"])
def test_invalid_data_names(name):
    with pytest.raises(ValueError):
        resolve_data_name(AppConfig(results_data_name=name))


def test_repeated_runs_are_isolated_but_monitor_is_stable(tmp_path):
    cfg = AppConfig(results_dir=str(tmp_path), model_name="naive", monitor_enabled=True)
    first = prepare_run_artifacts(cfg)
    second = prepare_run_artifacts(cfg)
    assert first.forecast_results_dir != second.forecast_results_dir
    assert first.monitor_dir == second.monitor_dir
    assert first.experiment_path == second.experiment_path


def test_path_components_are_bounded(tmp_path):
    cfg = AppConfig(results_dir=str(tmp_path), model_params={"long": "x" * 1000})
    paths = prepare_run_artifacts(cfg)
    assert paths.forecast_results_dir.is_dir()
    assert all(len(part.encode()) <= 180 for part in paths.experiment_path.parts)


def test_short_eda_path_preserves_identity_and_run_isolation(tmp_path):
    cfg = AppConfig(results_dir=str(tmp_path), freq="15min", do_eda=True,
                    do_train=False, do_test=False, do_forecast=False)
    first = prepare_run_artifacts(cfg, "first")
    assert len(first.eda_path.parts) == 1
    assert first.eda_path.name.startswith("15min_identity-v2-")
    assert (first.eda_dir.parent.parent / "identity.json").is_file()
    second = prepare_run_artifacts(cfg, "second")
    assert first.eda_dir != second.eda_dir and first.eda_path == second.eda_path
    limited = replace(cfg, eda_bds_max_samples=1000)
    assert build_eda_path(limited) != first.eda_path
    assert build_experiment_path(limited) == build_experiment_path(cfg)
