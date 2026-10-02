"""EDA 数据视图适配：验证而不修复，不隐式改变观测口径。"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from data_provider.quality.checks import require_finite, require_regular_time


def prepare_series(df: pd.DataFrame, time_col: str = "ds", target_col: str = "y", freq: str = "D") -> pd.Series:
    if time_col not in df or target_col not in df:
        raise ValueError("EDA input requires explicit time and target columns")
    times = pd.DatetimeIndex(pd.to_datetime(df[time_col]))
    values = df[target_col].astype(float)
    require_regular_time(times, freq=freq, role="EDA input")
    require_finite(values, role="EDA input")
    if len(values) < 10:
        raise ValueError("series is too short for reliable EDA (need >= 10 samples)")
    return pd.Series(values.to_numpy(), index=times, name=target_col)


def load_comparison_series(path: str | Path, time_col: str, target_col: str) -> pd.Series:
    """严格读取比较序列，不做补频或插值。"""
    frame = pd.read_csv(path)
    missing = [column for column in (time_col, target_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"EDA comparison columns not found in {path}: {missing}")
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise")
    if frame[time_col].isna().any() or frame[target_col].isna().any():
        raise ValueError(f"EDA comparison data contains missing values: {path}")
    return frame.sort_values(time_col).set_index(time_col)[target_col]
