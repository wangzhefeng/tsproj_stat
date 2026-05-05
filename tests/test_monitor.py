import pandas as pd

from evaluation.monitor import ModelMonitor
from run import run_monitor_actuals_backfill


def test_model_monitor_logs_actuals_and_metrics(tmp_path):
    monitor = ModelMonitor(tmp_path, setting="naive-demo-direct", window=3)
    yhat = pd.Series([10.0, 11.0, 12.0])

    monitor.log_forecast(run_id="run-1", yhat=yhat, forecast_ts="2026-05-04T00:00:00Z")
    monitor.fill_actuals(
        pd.Series([9.0, 11.5, 13.0]),
        forecast_ts="2026-05-04T00:00:00Z",
    )
    metrics = monitor.compute_rolling_metrics()
    snapshot = monitor.snapshot_metrics(run_id="run-1")

    assert metrics["mae"] > 0
    assert snapshot["mae"] == metrics["mae"]
    assert (tmp_path / "naive-demo-direct" / "predictions_log.csv").exists()
    assert (tmp_path / "naive-demo-direct" / "actuals_log.csv").exists()
    assert (tmp_path / "naive-demo-direct" / "metrics_history.csv").exists()


def test_model_monitor_fills_actuals_from_frame(tmp_path):
    monitor = ModelMonitor(tmp_path, setting="naive-demo-direct", window=3)
    monitor.log_forecast(
        run_id="run-1",
        yhat=pd.Series([10.0, 11.0, 12.0]),
        forecast_ts="2026-05-04T00:00:00Z",
    )

    monitor.fill_actuals_frame(
        pd.DataFrame(
            {
                "forecast_ts": ["2026-05-04T00:00:00Z"] * 3,
                "horizon_step": [1, 2, 3],
                "actual": [10.5, 10.0, 13.0],
            }
        ),
        actual_col="actual",
    )

    actuals = pd.read_csv(tmp_path / "naive-demo-direct" / "actuals_log.csv")
    assert actuals["y_true"].tolist() == [10.5, 10.0, 13.0]
    assert monitor.compute_rolling_metrics()["mae"] > 0


def test_monitor_actuals_backfill_cli_helper_writes_actuals_and_snapshot(tmp_path):
    monitor = ModelMonitor(tmp_path, setting="naive-demo-direct", window=3)
    forecast_ts = "2026-05-04T00:00:00Z"
    monitor.log_forecast(run_id="run-1", yhat=pd.Series([10.0, 11.0, 12.0]), forecast_ts=forecast_ts)
    actuals_path = tmp_path / "actuals.csv"
    actuals_path.write_text(
        "forecast_ts,horizon_step,actual\n"
        f"{forecast_ts},1,10.5\n"
        f"{forecast_ts},2,10.0\n"
        f"{forecast_ts},3,13.0\n",
        encoding="utf-8",
    )

    result = run_monitor_actuals_backfill(
        monitor_dir=str(tmp_path),
        setting="naive-demo-direct",
        actuals_path=str(actuals_path),
        actual_col="actual",
        snapshot=True,
        run_id="run-1",
        window=3,
    )

    assert result["monitor_actuals_path"].endswith("actuals_log.csv")
    assert result["monitor_metrics_path"].endswith("metrics_history.csv")
    assert result["metrics"]["mae"] > 0
