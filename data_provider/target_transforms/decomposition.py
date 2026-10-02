"""季节分解计算，不保存模型状态或执行未来外推。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decompose(series: pd.Series, period: int, method: str, model: str) -> tuple[pd.Series, pd.Series]:
    """执行单周期季节分解，返回 (trend, seasonal) 两个分量。

    method 支持 stl（robust）与 seasonal_decompose（model=additive/multiplicative）；
    两端边界通过 interpolate 补齐，保证分量与输入等长。
    """
    if method not in {"stl", "seasonal_decompose"}:
        raise ValueError("decomposition method must be stl or seasonal_decompose")
    if model not in {"additive", "multiplicative"}:
        raise ValueError("decomposition model must be additive or multiplicative")
    if method == "stl" and model != "additive":
        raise ValueError("STL requires additive decomposition")
    if isinstance(period, bool) or not isinstance(period, int) or period < 2:
        raise ValueError("seasonal period must be an integer > 1")
    if len(series) < period * 2:
        raise ValueError("decomposition requires two complete seasonal cycles")
    values = series.reset_index(drop=True)
    if method == "stl":
        from statsmodels.tsa.seasonal import STL

        result = STL(values, period=period, robust=True).fit()
        trend = pd.Series(result.trend).interpolate(limit_direction="both")
        seasonal = pd.Series(result.seasonal).interpolate(limit_direction="both")
        return trend, seasonal

    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(
        values,
        period=period,
        model=model,
        extrapolate_trend=period - 1,  # statsmodels 对 "freq" 的等价展开。
    )
    trend = pd.Series(result.trend).interpolate(limit_direction="both")
    seasonal = pd.Series(result.seasonal).interpolate(limit_direction="both")
    return trend, seasonal


def decompose_mstl(series: pd.Series, periods: list[int]) -> tuple[pd.Series, np.ndarray]:
    """计算加法多季节分解；保留所有周期，样本不足显式失败。"""
    from statsmodels.tsa.seasonal import MSTL

    if len(series) <= 2 * max(periods):
        raise ValueError("MSTL history must exceed twice the largest seasonal period")
    result = MSTL(series, periods=periods, stl_kwargs={"robust": True}).fit()
    components = np.asarray(result.seasonal)
    if components.ndim == 1:
        components = components[:, None]
    return pd.Series(result.trend).reset_index(drop=True), components
