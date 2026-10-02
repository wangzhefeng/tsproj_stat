"""离线季节槽补缺；双向观测，不作为训练窗口默认修复策略。"""
import numpy as np
import pandas as pd


def seasonal_slot_fill(series: pd.Series, weeks: int) -> pd.Series:
    """用局部周窗口中相同星期和时刻的观测均值填充缺失点。"""
    missing = series.isna().to_numpy()
    if not missing.any():
        return series

    index = series.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("seasonal_slot requires a DatetimeIndex")
    day_of_week = index.dayofweek.to_numpy()
    minute_of_day = (index.hour * 60 + index.minute).to_numpy()
    values = series.to_numpy(dtype=float)
    filled = series.copy()

    for raw_position in np.flatnonzero(missing):
        position = int(raw_position)
        timestamp = index[position]
        start = index.searchsorted(timestamp - pd.Timedelta(weeks=weeks), side="left")
        end = index.searchsorted(timestamp + pd.Timedelta(weeks=weeks), side="right")
        window = values[start:end]
        candidates = (
            (day_of_week[start:end] == day_of_week[position])
            & (minute_of_day[start:end] == minute_of_day[position])
            & ~np.isnan(window)
        )
        if candidates.any():
            filled.iloc[position] = float(window[candidates].mean())
    return filled
