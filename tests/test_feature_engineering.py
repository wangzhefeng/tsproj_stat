"""features.feature_engineering 单元测试：数值行为与 warmup 契约。"""
import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import FeatureEngineer, build_history_features


def _frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=n, freq="h"),
        "y": np.arange(n, dtype=float) + 1.0,
    })


def test_build_history_features_returns_warmup_and_columns():
    df = _frame()
    frame, columns, warmup = build_history_features(df, "ds", "y", True, [1, 3])
    assert warmup == 3
    assert {"hour", "dayofweek", "month", "dayofyear", "lag_1", "lag_3"} == set(columns)
    assert len(frame) == len(df)
    # lag_k 数值 = 目标列 shift(k)，warmup 头部为 NaN，之后完整
    assert frame["lag_1"].iloc[1] == df["y"].iloc[0]
    assert frame["lag_3"].iloc[3] == df["y"].iloc[0]
    assert frame["lag_3"].iloc[:3].isna().all()
    assert frame["lag_3"].iloc[3:].notna().all()


def test_build_history_features_empty_lags_zero_warmup():
    frame, columns, warmup = build_history_features(_frame(), "ds", "y", False, [])
    assert warmup == 0
    assert columns == []
    assert len(frame.columns) == 0


def test_build_history_features_rejects_non_positive_lag():
    with pytest.raises(ValueError, match="positive"):
        build_history_features(_frame(), "ds", "y", True, [0])


def test_create_features_snapshot_no_nan_and_shift_values():
    df = _frame(12)
    engineer = FeatureEngineer("ds", "y")
    out, feature_cols, target_shift_cols = engineer.create_features(
        df, enable_datetime_features=True, lags=[1, 2], horizon=2,
    )
    assert target_shift_cols == ["target_t_plus_1", "target_t_plus_2"]
    assert "lag_1" in feature_cols and "lag_2" in feature_cols
    # dropna 快照语义：整表无缺失；头部丢 warmup(max lag=2) 行、尾部丢 horizon=2 行
    assert not out.isna().any().any()
    assert len(out) == len(df) - 2 - 2
    # 未来标签数值 = 目标列向前 shift
    assert out["target_t_plus_1"].iloc[0] == df["y"].iloc[3]
