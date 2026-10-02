import json
from pathlib import Path

import pandas as pd
import pytest

from config import AppConfig
from pipeline.runner import ModelApp


def config(tmp_path, **kwargs):
    return AppConfig(results_dir=str(tmp_path), model_name="naive", history_size=30,
                     predict_horizon=2, do_eda=False, do_train=True, do_test=False, **kwargs)


def test_runs_isolate_outputs_and_publish_verified_manifest(tmp_path):
    first = ModelApp(config(tmp_path)).run()
    second = ModelApp(config(tmp_path)).run()
    assert first["prediction_path"] != second["prediction_path"]
    for result in (first, second):
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        assert manifest["status"] == "succeeded"
        assert manifest["stages"]["train"] == "succeeded"
        assert manifest["stages"]["test"] == "skipped"
        assert manifest["inputs"]["history_view"]["rows"] > 30
        from artifacts.manifest import verify_manifest
        verify_manifest(Path(result["manifest_path"]), tmp_path)
        prediction = pd.read_csv(result["prediction_path"])
        assert prediction.yhat.nunique() == 1
        assert (pd.to_datetime(prediction.timestamp).diff().dropna() == pd.Timedelta(days=1)).all()


def test_stage_failure_is_not_success(tmp_path, monkeypatch):
    app = ModelApp(config(tmp_path))
    def fail(*args):
        raise ValueError("controlled forecast failure")
    monkeypatch.setattr(app, "forecast", fail)
    out = app.run()
    manifest = json.loads(Path(out["manifest_path"]).read_text())
    assert manifest["status"] == "failed"
    assert manifest["stages"]["forecast"] == "failed"
    assert manifest["errors"]["forecast_error"] == "controlled forecast failure"


def test_load_failure_publishes_failed_manifest(tmp_path, monkeypatch):
    app = ModelApp(config(tmp_path))
    def fail():
        raise ValueError("controlled load failure")
    monkeypatch.setattr(app, "_load_dataset", fail)
    with pytest.raises(ValueError, match="controlled"):
        app.run()
    manifests = list(tmp_path.rglob("run_manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["status"] == "failed"


def test_missing_required_file_cannot_publish_success(tmp_path, monkeypatch):
    app = ModelApp(config(tmp_path))
    monkeypatch.setattr(app, "forecast", lambda _: {"prediction_path": str(tmp_path / "absent.csv")})
    result = app.run()
    assert "forecast_error" in result
    assert json.loads(Path(str(result["manifest_path"])).read_text())["status"] == "failed"


def test_backtest_plot_is_not_an_error_and_checksum_detects_change(tmp_path):
    from artifacts.manifest import verify_manifest
    cfg = config(tmp_path)
    cfg.do_test = True
    result = ModelApp(cfg).run()
    manifest = Path(str(result["manifest_path"]))
    assert verify_manifest(manifest, tmp_path)["status"] == "succeeded"
    target = Path(str(result["prediction_path"]))
    target.write_text(target.read_text() + "corrupt")
    with pytest.raises(ValueError, match="integrity"):
        verify_manifest(manifest, tmp_path)


def test_multi_model_partial_failure_has_failed_top_and_child(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.model_names = ["naive", "historic_average"]
    app = ModelApp(cfg)
    original = app.forecast
    def forecast(prepared):
        if app.cfg.model_name == "historic_average":
            raise ValueError("second model failed")
        return original(prepared)
    monkeypatch.setattr(app, "forecast", forecast)
    result = app.run()
    assert "run_error" in result
    top = json.loads(Path(str(result["manifest_path"])).read_text())
    assert top["status"] == "failed"
    for name, status in (("naive", "succeeded"), ("historic_average", "failed")):
        child = result[f"model::{name}"]
        payload = json.loads(Path(child["manifest_path"]).read_text())
        assert payload["status"] == status
        assert payload["run_id"] == top["run_id"]


def test_simulation_toggle_and_train_only_manifest(tmp_path):
    cfg = config(tmp_path, simulate_enabled=True, simulate_n_windows=3, simulate_n_paths=5)
    on = ModelApp(cfg).run()
    assert Path(str(on["simulated_paths_path"])).is_file()
    cfg.simulate_enabled = False
    off = ModelApp(cfg).run()
    manifest = json.loads(Path(str(off["manifest_path"])).read_text())
    assert not any("simulated_" in entry["path"] for entry in manifest["files"])
    cfg.do_forecast = False
    train_only = ModelApp(cfg).run()
    payload = json.loads(Path(str(train_only["manifest_path"])).read_text())
    assert payload["status"] == "succeeded"
    assert payload["stages"]["forecast"] == "skipped"


def test_interval_and_simulation_both_publish_valid_outputs(tmp_path):
    from artifacts.manifest import verify_manifest
    import numpy as np
    import pandas as pd
    cfg = config(tmp_path, simulate_enabled=True, simulate_n_windows=3, simulate_n_paths=5,
                 return_intervals=True, interval_method="conformal", conformal_n_windows=3,
                 interval_alpha=0.5, simulate_quantiles=[0.1, 0.9])
    out = ModelApp(cfg).run()
    assert "simulated_paths_path" in out, out
    point = pd.read_csv(str(out["prediction_path"]))
    assert {"yhat_lower", "yhat_upper"} <= set(point.columns)
    paths = pd.read_csv(str(out["simulated_paths_path"]))
    quantiles = pd.read_csv(str(out["simulated_quantiles_path"]))
    values = paths.value.to_numpy().reshape(5, cfg.predict_horizon)
    np.testing.assert_allclose(quantiles[["q10", "q90"]].to_numpy(), np.quantile(values, [0.1, 0.9], axis=0).T)
    manifest = verify_manifest(Path(str(out["manifest_path"])), tmp_path)
    assert manifest["status"] == "succeeded"


def test_manifest_cannot_succeed_when_requested_simulation_is_missing(tmp_path, monkeypatch):
    import pipeline.runner as runner
    original = runner.run_forecast_stage

    def lose_paths(*args, **kwargs):
        result = original(*args, **kwargs)
        result.simulate_paths_df = None
        return result

    monkeypatch.setattr(runner, "run_forecast_stage", lose_paths)
    out = ModelApp(config(tmp_path, simulate_enabled=True, simulate_n_windows=3, simulate_n_paths=5)).run()
    manifest = json.loads(Path(str(out["manifest_path"])).read_text())
    assert manifest["status"] == "failed"
    assert "simulated_paths_path" in manifest["errors"]["forecast_error"]
