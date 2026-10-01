"""EDA 数据视图适配：验证而不修复，不隐式改变观测口径。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def prepare_series(df: pd.DataFrame, time_col: str = "ds", target_col: str = "y", freq: str = "D") -> pd.Series:
    if time_col not in df or target_col not in df:
        raise ValueError("EDA input requires explicit time and target columns")
    times = pd.DatetimeIndex(pd.to_datetime(df[time_col]))
    values = df[target_col].to_numpy(dtype=float)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("EDA input requires sorted unique nonmissing timestamps")
    if not np.isfinite(values).all():
        raise ValueError("EDA input contains missing values; provide an explicitly repaired analysis view")
    if len(times) and not times.equals(pd.date_range(times[0], periods=len(times), freq=freq)):
        raise ValueError("EDA input is not regular; prepare an explicit resampled analysis view")
    if len(values) < 10:
        raise ValueError("series is too short for reliable EDA (need >= 10 samples)")
    return pd.Series(values, index=times, name=target_col)
