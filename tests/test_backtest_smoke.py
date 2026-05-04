import pandas as pd

from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory


def test_backtest_smoke():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=60, freq="D"), "y": list(range(60))})
    result = rolling_backtest(
        df,
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=30,
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
    assert result.summary["window_mode"] == "expanding"
    assert result.summary["inference_strategy"] == "direct"


def test_backtest_verbose_progress_logs(capsys):
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=60, freq="D"), "y": list(range(60))})

    rolling_backtest(
        df,
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=30,
        horizon=5,
        step=5,
        verbose=True,
        progress_every=2,
    )

    output = capsys.readouterr().out

    assert "[backtest]" in output
    assert "window" in output


def test_backtest_silent_by_default(capsys):
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=60, freq="D"), "y": list(range(60))})

    rolling_backtest(
        df,
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=30,
        horizon=5,
        step=5,
    )

    output = capsys.readouterr().out

    assert output == ""


def test_backtest_sliding_window_keeps_fixed_train_size():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=40, freq="D"), "y": list(range(40))})
    result = rolling_backtest(
        df,
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=10,
        horizon=5,
        step=5,
        window_mode="sliding",
        inference_strategy="recursive",
    )

    assert (result.metrics_df["train_end"] - result.metrics_df["train_start"]).eq(10).all()
    assert result.summary["window_mode"] == "sliding"
    assert result.summary["inference_strategy"] == "recursive"
