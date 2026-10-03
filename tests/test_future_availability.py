"""未来外生只使用预测原点当时可用的版本。"""
import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory


def test_unknown_future_truth_is_rejected_by_default():
    cfg = AppConfig(future_exog_path="forecast.csv", future_exog_time_col="ds", future_exog_cols=["x"])
    with pytest.raises(ValueError, match="known|issue"):
        cfg.validate()
    f = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=8),
                      "y": np.arange(8.), "x": np.arange(8.)})
    with pytest.raises(ValueError, match="known|as-of"):
        rolling_backtest(f, lambda: ModelFactory().create_model("linear_var"), time_col="ds",
                         exog_cols=["x"], future_exog_cols=["x"], train_size=5, horizon=3)


def test_forecast_vintages_ignore_late_publications_and_require_coverage():
    from data_provider.availability import FutureExogSource

    rows = pd.DataFrame({"valid": pd.to_datetime(["2026-01-03", "2026-01-04"] * 2),
                         "issued": pd.to_datetime(["2026-01-02"] * 2 + ["2026-01-03"] * 2),
                         "x": [3., 4., 1000., 2000.]})
    source = FutureExogSource(rows, "valid", ("x",), "issued")
    times = pd.date_range("2026-01-03", periods=2)
    pd.testing.assert_frame_equal(source.at(pd.Timestamp("2026-01-02"), times), pd.DataFrame({"x": [3., 4.]}))
    with pytest.raises(ValueError, match="available|cover"):
        source.at(pd.Timestamp("2026-01-01"), times)


def test_vintages_reach_backtest_conformal_simulation_and_final_forecast(tmp_path):
    from pipeline.runner import ModelApp

    dates = pd.date_range("2026-01-01", periods=47)
    x = np.random.default_rng(42).normal(size=47)
    history = pd.DataFrame({"ds": dates[:45], "y": 2 * x[:45] + 3, "x": x[:45]})
    early = pd.DataFrame({"valid": dates, "issued": pd.Timestamp("2025-12-31"), "x": x + .5})
    late = early.assign(issued=pd.Timestamp("2027-01-01"), x=1e6)
    cfg = AppConfig(model_name="linear_var", model_params={"target_lags": [1], "feature_lags": [0]},
                    exog_cols=["x"], future_exog_cols=["x"], future_exog_time_col="valid",
                    future_exog_issue_time_col="issued", history_size=30, predict_horizon=2,
                    backtest_train_size=30, backtest_horizon=2, backtest_step=15,
                    forecast_strategy="native", return_intervals=True, interval_method="conformal",
                    interval_alpha=.2, conformal_n_windows=4, simulate_enabled=True,
                    simulate_n_windows=4, simulate_n_paths=5, results_dir=str(tmp_path))
    cfg.validate(future_exog_available=True)
    out = ModelApp(cfg, data_frame=history, future_exog_frame=pd.concat([early, late])).run()
    assert not {k: v for k, v in out.items() if k.endswith("_error")}
    forecast_path, backtest_path, paths_path = out["prediction_path"], out["backtest_predictions_path"], out["simulated_paths_path"]
    assert isinstance(forecast_path, str) and isinstance(backtest_path, str) and isinstance(paths_path, str)
    forecast = pd.read_csv(forecast_path)
    np.testing.assert_allclose(forecast.yhat, 2 * (x[45:] + .5) + 3, atol=1e-8)
    np.testing.assert_allclose(forecast.yhat - forecast.yhat_lower, 1., atol=1e-8)
    backtest = pd.read_csv(backtest_path)
    np.testing.assert_allclose(backtest.y_pred, 2 * (x[30:32] + .5) + 3, atol=1e-8)
    paths = pd.read_csv(paths_path)
    np.testing.assert_allclose(paths.value.to_numpy().reshape(5, 2), np.tile(2 * x[45:] + 3, (5, 1)), atol=1e-8)
