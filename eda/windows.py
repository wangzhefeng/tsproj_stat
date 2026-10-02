"""月度/滚动画像、局部离线异常与连续事件；只计算，不读写或修复数据。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence import validate_periods


_STAT_COLUMNS = ["start", "end", "n_samples", "mean", "std", "min", "max", "slope_per_point"]
_PERIOD_COLUMNS = ["start", "end", "period", "status", "stable", "reason", "forward_r2", "backward_r2", "profile_correlation"]


def _statistics(series: pd.Series) -> dict:
    values = series.to_numpy(dtype=float)
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else None
    return dict(start=str(series.index[0]), end=str(series.index[-1]), n_samples=len(values),
                mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values)>1 else None,
                min=float(np.min(values)), max=float(np.max(values)), slope_per_point=slope)


def summarize_windows(series: pd.Series, periods: list[int], window_size: int, step: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = []
    if isinstance(series.index, pd.DatetimeIndex):
        for month, group in series.groupby(series.index.strftime("%Y-%m")):
            monthly.append({"month": month, **_statistics(group)})
    rolling, evidence = [], []
    if window_size:
        if window_size < 10 or not 1 <= step <= window_size:
            raise ValueError("invalid rolling window/step")
        if len(series) >= window_size:
            starts = sorted(set([*range(0, len(series)-window_size+1, step), len(series)-window_size]))
            for start in starts:
                view = series.iloc[start:start+window_size]
                stats = _statistics(view)
                rolling.append(stats)
                for row in validate_periods(view, periods):
                    evidence.append({"start": stats["start"], "end": stats["end"], **row})
    return (pd.DataFrame(monthly, columns=["month", *_STAT_COLUMNS]),
            pd.DataFrame(rolling, columns=_STAT_COLUMNS), pd.DataFrame(evidence, columns=_PERIOD_COLUMNS))


def local_outliers(series: pd.Series, residual: pd.Series, window: int) -> pd.DataFrame:
    if window < 3 or window % 2 == 0:
        raise ValueError("local window must be odd >= 3")
    rolling = residual.rolling(window, center=True, min_periods=window//2+1)
    center = rolling.median()
    mad = rolling.apply(lambda v: float(np.median(np.abs(v-np.median(v)))), raw=True)
    floor = np.finfo(float).eps * max(1., float(series.abs().max())) * 100
    scale = (1.4826 * mad).clip(lower=floor)
    lower, upper = center - 3*scale, center + 3*scale
    mask = (residual < lower) | (residual > upper)
    return pd.DataFrame({"time": [str(t) for t in series.index[mask]],
                         "value": series[mask].to_numpy(), "method": "local_residual_mad",
                         "tested_value": residual[mask].to_numpy(),
                         "lower": lower[mask].to_numpy(), "upper": upper[mask].to_numpy()})


def summarize_events(outliers: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    columns = ["method", "start", "end", "n_points", "duration_seconds", "peak_time", "peak_value", "max_deviation"]
    rows = []
    positions = {str(t): i for i,t in enumerate(series.index)}
    for method, group in outliers.groupby("method", sort=True):
        ordered = group.copy()
        ordered["position"] = ordered.time.map(positions)
        ordered = ordered.sort_values("position")
        groups = ordered.position.diff().ne(1).cumsum()
        for _, event in ordered.groupby(groups):
            first, last = int(event.position.iloc[0]), int(event.position.iloc[-1])
            deviations = np.maximum(event.lower.to_numpy()-event.tested_value.to_numpy(),
                                    event.tested_value.to_numpy()-event.upper.to_numpy())
            peak = event.iloc[int(np.argmax(deviations))]
            duration = None
            if isinstance(series.index, pd.DatetimeIndex):
                end_exclusive = (series.index[last+1] if last+1 < len(series) else
                                 series.index[-1]+(series.index[-1]-series.index[-2]))
                duration = (end_exclusive-series.index[first]).total_seconds()
            rows.append(dict(method=method, start=str(series.index[first]), end=str(series.index[last]),
                             n_points=len(event), duration_seconds=duration, peak_time=peak.time,
                             peak_value=float(peak.value), max_deviation=float(np.max(deviations))))
    return pd.DataFrame(rows, columns=columns)
