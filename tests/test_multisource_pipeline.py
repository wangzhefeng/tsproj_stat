from pathlib import Path

import pandas as pd

from app import ModelApp
from config import AppConfig


def test_pipeline_multisource_linear_var_forecast_contract(tmp_path):
    history_path = tmp_path / "history.csv"
    future_path = tmp_path / "future.csv"

    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=24, freq="D"),
            "y": [float(v) for v in range(24)],
            "load": [float(v * 2) for v in range(24)],
            "temp": [15.0 + (v % 5) for v in range(24)],
        }
    )
    future = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-25", periods=4, freq="D"),
            "temp": [18.0, 19.0, 20.0, 21.0],
        }
    )
    history.to_csv(history_path, index=False)
    future.to_csv(future_path, index=False)

    cfg = AppConfig(
        data_path=str(history_path),
        time_col="ds",
        target_col="y",
        model_name="linear_var",
        model_params={"target_lags": [1, 2], "feature_lags": [0, 1]},
        endog_cols=["y", "load"],
        hist_exog_cols=["temp"],
        future_exog_path=str(future_path),
        future_exog_time_col="ds",
        future_exog_cols=["temp"],
        do_eda=False,
        do_train=True,
        do_test=True,
        do_forecast=True,
        history_size=12,
        predict_horizon=4,
        backtest_initial_train_size=12,
        backtest_horizon=4,
        backtest_step=4,
        checkpoints_dir=str(tmp_path / "saved_results" / "checkpoints"),
        train_results_dir=str(tmp_path / "saved_results" / "results_train"),
        test_results_dir=str(tmp_path / "saved_results" / "results_test"),
        forecast_result_dir=str(tmp_path / "saved_results" / "results_forecast"),
        eda_output_dir=str(tmp_path / "saved_results" / "results_eda"),
    )

    result = ModelApp(cfg).run()
    forecast_df = pd.read_csv(result["prediction_path"])

    assert list(forecast_df.columns) == ["step", "timestamp", "yhat"]
    assert len(forecast_df) == 4
    assert Path(result["backtest_predictions_path"]).exists()
