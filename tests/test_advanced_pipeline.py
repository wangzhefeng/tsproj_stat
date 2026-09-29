import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.runner import ModelApp
from config import AppConfig


def test_conformal_pipeline_outputs_original_scale_and_backtest_metrics(tmp_path):
    source = tmp_path / "history.csv"
    pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=50), "y": np.arange(100., 150.)}).to_csv(source, index=False)
    cfg = AppConfig(data_path=str(source), model_name="naive", forecast_strategy="native",
                    results_dir=str(tmp_path / "results"), history_size=20, predict_horizon=2,
                    backtest_train_size=20, backtest_horizon=2, backtest_step=10,
                    return_intervals=True, interval_method="conformal", conformal_n_windows=4,
                    interval_alpha=0.2, detrend_method="linear", do_train=True, do_test=True)
    cfg.validate()
    out = ModelApp(cfg).run()
    assert not [k for k in out if k.endswith("_error")]
    forecast = pd.read_csv(out["prediction_path"])
    np.testing.assert_allclose(forecast.yhat, [150., 151.], atol=1e-8)
    np.testing.assert_allclose(forecast.yhat_lower, [150., 151.], atol=1e-8)
    summary = json.loads(Path(out["forecast_summary_path"]).read_text())
    assert summary["interval_method"] == "conformal"
    metrics = pd.read_csv(out["test_metrics_path"])
    assert metrics.interval_width.max() < 1e-8
    assert Path(out["model_path"]).is_file()


def test_new_cli_fields_roundtrip(monkeypatch, tmp_path):
    from run import parse_args
    monkeypatch.setattr("sys.argv", ["run.py", "--results_dir", str(tmp_path),
        "--forecast_strategy", "native", "--decomposition_method", "mstl", "--seasonal_periods", "7,24",
        "--interval_method", "conformal", "--conformal_n_windows", "4", "--interval_alpha", "0.2"])
    cfg = parse_args()
    assert cfg.seasonal_periods == [7, 24]
    assert cfg.interval_method == "conformal"
    assert cfg.conformal_n_windows == 4
