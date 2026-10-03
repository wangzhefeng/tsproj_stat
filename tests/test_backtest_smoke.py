import pandas as pd
import pytest

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
    assert result.summary["forecast_strategy"] == "direct"


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
        forecast_strategy="recursive",
    )

    assert (result.metrics_df["train_end"] - result.metrics_df["train_start"]).eq(10).all()
    assert result.summary["window_mode"] == "sliding"
    assert result.summary["forecast_strategy"] == "recursive"


def test_backtest_parallel_matches_serial_output():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=60, freq="D"), "y": list(range(60))})
    kwargs = dict(
        df=df,
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=30,
        horizon=5,
        step=5,
    )

    serial = rolling_backtest(**kwargs, n_jobs=1)
    parallel = rolling_backtest(**kwargs, n_jobs=2)

    pd.testing.assert_frame_equal(serial.metrics_df, parallel.metrics_df)
    pd.testing.assert_frame_equal(serial.predictions_df, parallel.predictions_df)
    assert serial.summary == parallel.summary


def test_backtest_parallel_all_failed_windows_raise():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=40, freq="D"), "y": list(range(40))})

    def failing_builder():
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="All backtest windows failed"):
        rolling_backtest(
            df,
            model_builder=failing_builder,
            target_col="y",
            time_col="ds",
            train_size=10,
            horizon=5,
            step=5,
            n_jobs=2,
            allow_failed_windows=True,
        )


def _partially_failing_builder():
    """窗口 2 起拟合失败的模型构造器。

    direct 策略每窗口按 step 调用 horizon 次 builder（horizon=5），
    因此窗口 1 占用第 1–5 次调用，第 6 次起属于窗口 2。
    """
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("window fit failed")
        return ModelFactory().create_model("naive")

    return builder


def test_backtest_summary_discloses_future_exog_policy():
    """显式合成已知输入应披露 known_in_advance；默认未知输入由独立门禁测试拒绝。"""
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=60, freq="D"),
            "y": list(range(60)),
            "temp": [15.0 + (i % 5) for i in range(60)],
        }
    )
    result = rolling_backtest(
        df,
        model_builder=lambda: ModelFactory().create_model("linear_var", {"feature_lags": [0]}),
        target_col="y",
        time_col="ds",
        exog_cols=["temp"],
        future_exog_cols=["temp"],
        exog_future_known=True,
        train_size=30,
        horizon=5,
        step=5,
    )
    assert result.summary["future_exog_policy"] == "known_in_advance"

    result_plain = rolling_backtest(
        df[["ds", "y"]],
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=30,
        horizon=5,
        step=5,
    )
    assert result_plain.summary["future_exog_policy"] == "none"


def test_backtest_partial_failure_raises_by_default():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=40, freq="D"), "y": list(range(40))})

    with pytest.raises(RuntimeError, match="window 2/.*failed"):
        rolling_backtest(
            df,
            model_builder=_partially_failing_builder(),
            target_col="y",
            time_col="ds",
            train_size=10,
            horizon=5,
            step=5,
        )


def test_backtest_partial_failure_allowed_marks_survivor_bias():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=40, freq="D"), "y": list(range(40))})

    result = rolling_backtest(
        df,
        model_builder=_partially_failing_builder(),
        target_col="y",
        time_col="ds",
        train_size=10,
        horizon=5,
        step=5,
        allow_failed_windows=True,
    )

    assert len(result.failed_windows) > 0
    assert result.summary["survivor_bias"] is True or result.summary["survivor_bias"] == 1
    assert result.summary["failed_windows"] == len(result.failed_windows)
    assert len(result.metrics_df) == result.summary["window_count"]
