"""趋势拟合纯计算；未来还原参数由目标变换器持有。"""
import numpy as np
import pandas as pd


def fit_trend(series: pd.Series, method: str, window: int) -> tuple[pd.Series, float, float]:
    if method == "none":
        return pd.Series(np.zeros(len(series)), index=series.index), 0.0, 0.0
    if method == "linear":
        x = np.arange(len(series), dtype=float)
        if len(series) < 2:
            slope, intercept = 0.0, float(series.iloc[-1]) if len(series) else 0.0
        else:
            slope, intercept = np.polyfit(x, series.to_numpy(dtype=float), deg=1)
        return pd.Series(slope * x + intercept, index=series.index), float(slope), float(intercept)
    if method == "moving_average":
        return series.rolling(window=window, min_periods=1).mean(), 0.0, 0.0
    raise ValueError(f"unsupported trend method: {method}")
