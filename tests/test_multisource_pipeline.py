from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ModelApp
from config import AppConfig
from artifacts.checkpoints import load_model


def test_pipeline_multisource_linear_var_forecast_contract(tmp_path):
    history_path = tmp_path / "history.csv"
    future_path = tmp_path / "future.csv"

    temperature = np.random.default_rng(11).uniform(15.0, 25.0, 24)
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=24, freq="D"),
            "y": 5.0 + 2.0 * temperature,
            "load": [float(v * 2) for v in range(24)],
            "temp": temperature,
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
        model_params={"target_lags": [1], "feature_lags": [0]},
        endog_cols=["load"],
        exog_cols=["temp"],
        future_exog_path=str(future_path),
        future_exog_time_col="ds",
        future_exog_cols=["temp"],
        exog_future_known=True,  # 合成已知外生通路；真实天气预报另测 issue-time 档案。
        do_eda=False,
        do_train=True,
        do_test=True,
        do_forecast=True,
        history_size=12,
        predict_horizon=4,
        backtest_initial_train_size=12,
        backtest_horizon=4,
        backtest_step=4,
        results_dir=str(tmp_path / "results"),
    )

    result = ModelApp(cfg).run()
    forecast_df = pd.read_csv(result["prediction_path"])

    assert list(forecast_df.columns) == ["step", "timestamp", "yhat"]
    assert len(forecast_df) == 4
    expected = 5.0 + 2.0 * future["temp"]
    np.testing.assert_allclose(forecast_df["yhat"], expected, atol=1e-8, rtol=0)
    pd.testing.assert_series_equal(pd.to_datetime(forecast_df["timestamp"]), future["ds"], check_names=False)
    backtest = pd.read_csv(result["backtest_predictions_path"])
    assert not backtest.empty
    np.testing.assert_allclose(backtest["y_pred"], backtest["y_true"], atol=1e-8, rtol=0)
    # 归档不参与 forecast，但保存的训练快照仍应能独立预测。
    archived = load_model(result["model_path"])
    np.testing.assert_allclose(archived.predict(4, X_future=future.drop(columns="ds")), expected, atol=1e-8, rtol=0)
    assert Path(result["train_summary_path"]).is_file()
