import pandas as pd

from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory


def test_backtest_smoke():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=60, freq="D"), "y": list(range(60))})
    model = ModelFactory().create_model("naive")
    result = rolling_backtest(
        df,
        model=model,
        target_col="y",
        time_col="ds",
        initial_train_size=30,
        horizon=5,
        step=5,
    )

    assert len(result.metrics_df) > 0
    assert {"mae", "rmse", "mape", "smape", "mse", "r2", "bias", "max_error"}.issubset(
        set(result.metrics_df.columns)
    )
    assert {"window_id", "horizon_step", "y_true", "y_pred", "residual", "timestamp"}.issubset(
        set(result.predictions_df.columns)
    )
