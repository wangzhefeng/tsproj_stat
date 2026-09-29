import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig


def test_batch_isolates_series_and_models(tmp_path):
    from pipeline.panel import run_batch
    source = tmp_path / "panel.csv"
    pd.concat([pd.DataFrame({"id": sid, "ds": pd.date_range("2024-01-01", periods=30), "y": value})
               for sid, value in [("A/1", 3.), ("B", 20.)]], ignore_index=True).to_csv(source, index=False)
    cfg = AppConfig(data_path=str(source), series_id_col="id", batch_models={"naive": {}, "historic_average": {}},
                    results_dir=str(tmp_path / "results"), history_size=20, predict_horizon=3,
                    forecast_strategy="native", do_train=True, do_test=False)
    out = run_batch(cfg)
    manifest = json.loads(Path(out["batch_manifest_path"]).read_text())
    assert manifest["task_count"] == 4
    assert manifest["failed_count"] == 0
    paths = []
    for task in manifest["tasks"]:
        paths.append(task["artifacts"]["prediction_path"])
        actual = pd.read_csv(paths[-1]).yhat
        np.testing.assert_allclose(actual, 3. if task["series_id"] == "A/1" else 20.)
        assert Path(task["artifacts"]["model_path"]).is_file()
    assert len(set(paths)) == 4


def test_batch_failure_is_audited_and_not_success(tmp_path):
    from pipeline.panel import run_batch
    source = tmp_path / "short.csv"
    pd.DataFrame({"id": ["A"] * 3, "ds": pd.date_range("2024-01-01", periods=3), "y": [1., 2., 3.]}).to_csv(source, index=False)
    cfg = AppConfig(data_path=str(source), series_id_col="id", batch_models={"naive": {}},
                    results_dir=str(tmp_path / "results"), history_size=20, do_test=False)
    with pytest.raises(RuntimeError, match="batch.*failed"):
        run_batch(cfg)
    manifests = list((tmp_path / "results").rglob("batch_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["failed_count"] == 1
    assert manifest["tasks"][0]["status"] == "failed"


def test_batch_collection_failure_does_not_publish_partial_task(tmp_path, monkeypatch):
    from pipeline.panel import run_batch
    source = tmp_path / "panel.csv"
    pd.DataFrame({"id": "A", "ds": pd.date_range("2024-01-01", periods=30),
                  "y": np.arange(30.)}).to_csv(source, index=False)
    original = pd.read_csv

    def fail_metrics(path, *args, **kwargs):
        if Path(path).name == "backtest_metrics.csv":
            raise OSError("metrics read failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fail_metrics)
    cfg = AppConfig(data_path=str(source), series_id_col="id", batch_models={"naive": {}},
                    results_dir=str(tmp_path / "results"), history_size=20, batch_allow_failed=True)
    out = run_batch(cfg)
    manifest = json.loads(Path(out["batch_manifest_path"]).read_text())
    assert manifest["failed_count"] == 1
    assert manifest["tasks"][0]["status"] == "failed"
    assert "batch_predictions_path" not in out


def test_batch_parallel_matches_serial_results(tmp_path):
    """batch_n_jobs=2 进程池并行与串行结果一致（内存帧直通 + 任务分片）。"""
    from pipeline.panel import run_batch
    source = tmp_path / "panel.csv"
    pd.concat([pd.DataFrame({"id": sid, "ds": pd.date_range("2024-01-01", periods=30), "y": value})
               for sid, value in [("A/1", 3.), ("B", 20.), ("C", 7.)]], ignore_index=True).to_csv(source, index=False)

    def build(n_jobs):
        return AppConfig(data_path=str(source), series_id_col="id",
                         batch_models={"naive": {}, "historic_average": {}},
                         results_dir=str(tmp_path / f"results-{n_jobs}"), history_size=20,
                         predict_horizon=3, forecast_strategy="native", do_train=True,
                         do_test=False, batch_n_jobs=n_jobs)

    out_serial = run_batch(build(1))
    out_parallel = run_batch(build(2))
    serial = pd.read_csv(out_serial["batch_predictions_path"]).sort_values(["series_id", "model_name"]).reset_index(drop=True)
    parallel = pd.read_csv(out_parallel["batch_predictions_path"]).sort_values(["series_id", "model_name"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(serial, parallel)
    manifest = json.loads(Path(out_parallel["batch_manifest_path"]).read_text())
    assert manifest["failed_count"] == 0
    assert manifest["task_count"] == 6
