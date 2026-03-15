from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_series(
    df: pd.DataFrame,
    time_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
) -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")

    local = df.copy()
    if time_col in local.columns:
        local[time_col] = pd.to_datetime(local[time_col])
        local = local.sort_values(time_col).set_index(time_col)
    else:
        local.index = pd.date_range("2000-01-01", periods=len(local), freq=freq)

    series = pd.to_numeric(local[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    series = series.interpolate(limit_direction="both").dropna()

    if isinstance(series.index, pd.DatetimeIndex):
        series = series.asfreq(freq)
        series = series.interpolate(limit_direction="both").dropna()

    if len(series) < 10:
        raise ValueError("series is too short for reliable EDA (need >= 10 samples)")
    return series.rename(target_col)
