import argparse

import pandas as pd
import pytest

from config import AppConfig
from data_provider.data_loader import DataLoader
from run import _apply_overrides


def test_cli_override_multisource_fields():
    cfg = AppConfig()
    args = argparse.Namespace(
        project_name=None,
        seed=None,
        data_path=None,
        time_col=None,
        target_col=None,
        freq=None,
        model_name=None,
        model_params=None,
        pred_method=None,
        do_train=None,
        do_test=None,
        do_forecast=None,
        do_eda=None,
        history_size=None,
        predict_horizon=None,
        backtest_initial_train_size=None,
        backtest_horizon=None,
        backtest_step=None,
        backtest_verbose=None,
        backtest_progress_every=None,
        enable_datetime_features=None,
        lags=None,
        scale=None,
        scaler_type=None,
        denoise_enabled=None,
        denoise_window=None,
        detrend_method=None,
        checkpoints_dir=None,
        train_results_dir=None,
        test_results_dir=None,
        forecast_result_dir=None,
        eda_output_dir=None,
        endog_cols="y,load,price",
        exog_cols="temp,is_holiday",
        future_exog_path="/tmp/future_exog.csv",
        future_exog_time_col="forecast_time",
        future_exog_cols="temp,is_holiday",
    )

    updated = _apply_overrides(cfg, args)

    assert updated.endog_cols == ["y", "load", "price"]
    assert updated.exog_cols == ["temp", "is_holiday"]
    assert updated.future_exog_path == "/tmp/future_exog.csv"
    assert updated.future_exog_time_col == "forecast_time"
    assert updated.future_exog_cols == ["temp", "is_holiday"]


def test_data_loader_loads_future_exog_and_aligns_to_future_window(tmp_path):
    history_path = tmp_path / "history.csv"
    future_path = tmp_path / "future.csv"

    pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=8, freq="D"),
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
            "load": [10, 11, 12, 13, 14, 15, 16, 17],
            "temp": [20, 21, 22, 23, 24, 25, 26, 27],
        }
    ).to_csv(history_path, index=False)
    pd.DataFrame(
        {
            "forecast_time": pd.date_range("2024-01-09", periods=2, freq="D"),
            "temp": [28, 29],
            "is_holiday": [0, 1],
        }
    ).to_csv(future_path, index=False)

    loader = DataLoader(
        data_path=str(history_path),
        time_col="ds",
        target_col="y",
        value_cols=["y", "load", "temp"],
        future_exog_path=str(future_path),
        future_exog_time_col="forecast_time",
    )
    history_df = loader.load_data()
    future_exog_df = loader.load_future_exog(
        future_exog_cols=["temp", "is_holiday"],
        horizon=2,
    )

    assert list(history_df.columns) == ["ds", "y", "load", "temp"]
    assert list(future_exog_df.columns) == ["forecast_time", "temp", "is_holiday"]
    assert future_exog_df["temp"].tolist() == [28, 29]


def test_data_loader_future_exog_missing_required_columns_raises(tmp_path):
    future_path = tmp_path / "future.csv"
    pd.DataFrame(
        {
            "forecast_time": pd.date_range("2024-01-09", periods=2, freq="D"),
            "temp": [28, 29],
        }
    ).to_csv(future_path, index=False)

    loader = DataLoader(
        data_path=None,
        future_exog_path=str(future_path),
        future_exog_time_col="forecast_time",
    )

    with pytest.raises(ValueError, match="future_exog_cols"):
        loader.load_future_exog(future_exog_cols=["temp", "is_holiday"], horizon=2)
