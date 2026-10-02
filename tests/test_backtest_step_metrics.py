"""回测缩放指标（mase/rmsse）与按 horizon_step 聚合的 step_metrics_df。"""
import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import BacktestResult, rolling_backtest
from models.factory import ModelFactory


def _run_sliding() -> BacktestResult:
    df = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=40, freq="D"), "y": [float(i) for i in range(40)]}
    )
    return rolling_backtest(
        df,
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=10,
        horizon=3,
        step=3,
        window_mode="sliding",
    )


def test_backtest_reports_scaled_metrics_matching_point_metrics_on_unit_scale():
    # sliding 窗口下每个训练窗为 10 个连续整数，naive 差分恒为 1 → mase==mae、rmsse==rmse。
    result = _run_sliding()
    assert {"mase", "rmsse"}.issubset(set(result.metrics_df.columns))
    np.testing.assert_allclose(result.metrics_df["mase"], result.metrics_df["mae"])
    np.testing.assert_allclose(result.metrics_df["rmsse"], result.metrics_df["rmse"])
    assert result.summary["mase"] == pytest.approx(result.summary["mae"])
    assert result.summary["rmsse"] == pytest.approx(result.summary["rmse"])


def test_step_metrics_df_aggregates_per_horizon_step():
    result = _run_sliding()
    step_df = result.step_metrics_df
    # naive 预测 = 训练窗末值，y 恒等于行号 → 第 k 步绝对误差恒为 k（scale=1）。
    assert list(step_df["horizon_step"]) == [1, 2, 3]
    np.testing.assert_allclose(step_df["mae"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(step_df["rmse"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(step_df["mase"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(step_df["rmsse"], [1.0, 2.0, 3.0])
    assert (step_df["window_count"] == result.summary["window_count"]).all()


def test_scaled_metrics_nan_when_train_constant():
    # 常数序列训练窗缩放基准为 0 → mase/rmsse 为 NaN 且不打爆点指标。
    df = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=30, freq="D"), "y": [5.0] * 30}
    )
    result = rolling_backtest(
        df,
        model_builder=lambda: ModelFactory().create_model("naive"),
        target_col="y",
        time_col="ds",
        train_size=10,
        horizon=3,
        step=3,
    )
    assert np.isnan(result.summary["mase"])
    assert np.isnan(result.summary["rmsse"])
    assert result.summary["mae"] == 0.0
    assert result.step_metrics_df["mase"].isna().all()
