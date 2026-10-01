"""轻量去噪；移除的噪声不参与逆变换。"""
from __future__ import annotations

import pandas as pd


def remove_noise(series: pd.Series, method: str = "moving_average", window: int = 3) -> pd.Series:
    """执行轻量去噪；当前只保留无额外依赖的滑动均值和滑动中位数。"""
    if window < 1:
        raise ValueError("window must be >= 1")
    if method == "moving_average":
        return series.rolling(window=window, min_periods=1).mean()
    if method == "moving_median":
        if len(series) < window:
            return series.copy()
        return series.rolling(window=window, min_periods=1, center=True).median().bfill().ffill()
    if method == "none":
        return series.copy()
    raise ValueError("method must be one of {'none', 'moving_average', 'moving_median'}")
