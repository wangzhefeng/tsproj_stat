from pathlib import Path

import pandas as pd

from evaluation.visualization import (
    plot_backtest_predictions,
    plot_backtest_residuals,
    plot_error_distribution,
    plot_forecast,
)


def test_visualization_smoke(tmp_path):
    predictions_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
            "horizon_step": [1, 2, 3, 1, 2, 3],
            "y_true": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "y_pred": [1.1, 1.9, 3.2, 1.8, 2.8, 4.2],
            "residual": [-0.1, 0.1, -0.2, 0.2, 0.2, -0.2],
        }
    )
    history_df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
            "y": [float(i) for i in range(10)],
        }
    )
    forecast_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-11", periods=3, freq="D"),
            "yhat": [10.0, 11.0, 12.0],
        }
    )

    outputs = [
        plot_backtest_predictions(predictions_df, str(tmp_path / "pred.png"), "pred"),
        plot_backtest_residuals(predictions_df, str(tmp_path / "resid.png"), "resid"),
        plot_error_distribution(predictions_df, str(tmp_path / "hist.png"), "hist"),
        plot_forecast(history_df, forecast_df, str(tmp_path / "forecast.png"), "forecast", "ds", "y"),
    ]

    for output in outputs:
        assert Path(output).exists()
