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


def test_batch_resume_skips_succeeded_tasks(tmp_path):
    """断点续跑：上次成功任务跳过重跑、产物并入本次汇总；失败任务重跑。"""
    from pipeline.panel import run_batch
    source = tmp_path / "panel.csv"
    # B 序列只有 3 行：history_size=20 下必然失败 → 上次 failed，本次仍 failed
    pd.concat([pd.DataFrame({"id": sid, "ds": pd.date_range("2024-01-01", periods=n), "y": value})
               for sid, value, n in [("A", 3., 30), ("B", 20., 3)]], ignore_index=True).to_csv(source, index=False)

    def build(resume_from=None, results_dir=None):
        return AppConfig(data_path=str(source), series_id_col="id",
                         batch_models={"naive": {}},
                         results_dir=str(results_dir or tmp_path / "results"),
                         history_size=20, predict_horizon=3, forecast_strategy="native",
                         do_train=True, do_test=False, batch_allow_failed=True,
                         batch_resume_from=resume_from)

    first = run_batch(build())
    first_manifest = json.loads(Path(first["batch_manifest_path"]).read_text())
    assert first_manifest["task_count"] == 2 and first_manifest["failed_count"] == 1
    succeeded = next(t for t in first_manifest["tasks"] if t["status"] == "success")
    assert succeeded["series_id"] == "A"

    # 续跑：携带 A（不重跑），B 重跑后仍失败
    second = run_batch(build(resume_from=first["batch_manifest_path"]))
    second_manifest = json.loads(Path(second["batch_manifest_path"]).read_text())
    by_key = {(t["series_id"], t["model_name"]): t for t in second_manifest["tasks"]}
    assert by_key[("A", "naive")].get("resumed") is True
    assert by_key[("A", "naive")]["status"] == "success"
    assert "resumed" not in by_key[("B", "naive")]
    assert by_key[("B", "naive")]["status"] == "failed"
    # 携带任务的产物并入汇总（A 的预测仍在 batch_predictions.csv 中）
    predictions = pd.read_csv(second["batch_predictions_path"])
    assert (predictions["series_id"] == "A").all()
    np.testing.assert_allclose(predictions["yhat"], 3.)
    # 产物路径仍指向上次文件（不重生成）
    assert by_key[("A", "naive")]["artifacts"]["prediction_path"] == succeeded["artifacts"]["prediction_path"]


def test_batch_resume_rejects_config_mismatch(tmp_path):
    from pipeline.panel import run_batch
    source = tmp_path / "panel.csv"
    pd.DataFrame({"id": "A", "ds": pd.date_range("2024-01-01", periods=30),
                  "y": np.arange(30.)}).to_csv(source, index=False)

    def build(history_size=20, resume_from=None):
        return AppConfig(data_path=str(source), series_id_col="id", batch_models={"naive": {}},
                         results_dir=str(tmp_path / "results"), history_size=history_size,
                         predict_horizon=3, forecast_strategy="native", do_train=True,
                         do_test=False, batch_allow_failed=True, batch_resume_from=resume_from)

    first = run_batch(build())
    # history_size 变了 → RAISE
    with pytest.raises(ValueError, match="resume mismatch on 'history_size'"):
        run_batch(build(history_size=21, resume_from=first["batch_manifest_path"]))


def test_batch_resume_requires_series_id_col(tmp_path):
    cfg = AppConfig(data_path=str(tmp_path / "x.csv"), batch_resume_from="whatever.json")
    with pytest.raises(ValueError, match="batch_resume_from requires series_id_col"):
        cfg.validate()


def test_batch_resume_rejects_missing_manifest(tmp_path):
    from pipeline.panel import run_batch
    source = tmp_path / "panel.csv"
    pd.DataFrame({"id": "A", "ds": pd.date_range("2024-01-01", periods=30),
                  "y": np.arange(30.)}).to_csv(source, index=False)
    cfg = AppConfig(data_path=str(source), series_id_col="id", batch_models={"naive": {}},
                    results_dir=str(tmp_path / "results"), history_size=20, do_test=False,
                    batch_allow_failed=True, batch_resume_from=str(tmp_path / "nope.json"))
    with pytest.raises(ValueError, match="batch_resume_from path not found"):
        run_batch(cfg)


def _resume_config(tmp_path):
    source = tmp_path / "resume-input.csv"
    pd.concat([pd.DataFrame({"id": sid, "ds": pd.date_range("2024-01-01", periods=30), "y": value})
               for sid, value in [("A", 3.), ("B", 20.)]], ignore_index=True).to_csv(source, index=False)
    return AppConfig(data_path=str(source), series_id_col="id", batch_models={"naive": {}},
                     results_dir=str(tmp_path / "results"), history_size=20, predict_horizon=3,
                     forecast_strategy="native", do_train=True, do_test=False, batch_allow_failed=True)


def test_batch_resume_recovers_transient_failure(tmp_path, monkeypatch):
    from pipeline.panel import run_batch
    from pipeline.runner import ModelApp
    cfg = _resume_config(tmp_path)
    original = ModelApp.run

    def transient(app):
        if app.source_identity["series_id"] == "B":
            raise OSError("transient outage")
        return original(app)

    with monkeypatch.context() as patcher:
        patcher.setattr(ModelApp, "run", transient)
        first = run_batch(cfg)
    cfg.batch_resume_from = first["batch_manifest_path"]
    second = run_batch(cfg)
    manifest = json.loads(Path(second["batch_manifest_path"]).read_text())
    assert manifest["failed_count"] == 0
    tasks = {task["series_id"]: task for task in manifest["tasks"]}
    assert tasks["A"]["resumed"] is True
    assert tasks["B"]["status"] == "success" and not tasks["B"].get("resumed", False)
    prediction = pd.read_csv(tasks["B"]["artifacts"]["prediction_path"])
    np.testing.assert_allclose(prediction.yhat, 20.)


@pytest.mark.parametrize("damage", ["input", "artifact", "missing_config", "missing_fingerprint", "other_series", "missing_output"])
def test_batch_resume_requires_complete_matching_evidence(tmp_path, damage):
    from pipeline.panel import run_batch
    cfg = _resume_config(tmp_path)
    first = run_batch(cfg)
    path = Path(first["batch_manifest_path"])
    manifest = json.loads(path.read_text())
    if damage == "input":
        assert cfg.data_path is not None
        frame = pd.read_csv(cfg.data_path)
        frame["y"] += 100
        frame.to_csv(cfg.data_path, index=False)
    elif damage == "artifact":
        pred_path = Path(manifest["tasks"][0]["artifacts"]["prediction_path"])
        frame = pd.read_csv(pred_path)
        frame["yhat"] += 100
        frame.to_csv(pred_path, index=False)
    elif damage == "missing_config":
        manifest["config"].pop("history_size")
        path.write_text(json.dumps(manifest))
    elif damage == "other_series":
        manifest["tasks"][0]["artifacts"] = manifest["tasks"][1]["artifacts"]
        path.write_text(json.dumps(manifest))
    elif damage == "missing_output":
        manifest["tasks"][0]["artifacts"].pop("prediction_path")
        path.write_text(json.dumps(manifest))
    else:
        manifest.pop("inputs", None)
        path.write_text(json.dumps(manifest))
    cfg.batch_resume_from = str(path)
    with pytest.raises(ValueError, match="resume|integrity"):
        run_batch(cfg)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_batch_future_exog_in_memory(tmp_path, n_jobs):
    from pipeline.panel import run_batch
    cfg = _resume_config(tmp_path)
    assert cfg.data_path is not None
    history = pd.read_csv(cfg.data_path)
    x = np.random.default_rng(9).normal(size=len(history))
    history["x"], history["y"] = x, 2 * x + 1
    history.to_csv(cfg.data_path, index=False)
    future = pd.concat([pd.DataFrame({"id": sid, "ds": pd.date_range("2024-01-31", periods=3),
                                     "x": [11., 23., 37.]}) for sid in ("A", "B")])
    future_path = tmp_path / "future.csv"
    future.to_csv(future_path, index=False)
    cfg.batch_models = {"linear_var": {"target_lags": [1], "feature_lags": [0], "require_future_exog": True}}
    cfg.exog_cols = cfg.future_exog_cols = ["x"]
    cfg.future_exog_path, cfg.future_exog_time_col = str(future_path), "ds"
    cfg.batch_n_jobs = n_jobs
    out = run_batch(cfg)
    manifest = json.loads(Path(out["batch_manifest_path"]).read_text())
    assert manifest["failed_count"] == 0, manifest["tasks"]
    preds = pd.read_csv(out["batch_predictions_path"])
    for _, group in preds.groupby("series_id"):
        np.testing.assert_allclose(group.yhat, [23., 47., 75.], atol=1e-8)
