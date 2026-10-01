"""内存时间序列聚合：不读写文件、不依赖应用配置。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AGGREGATION_METHODS = {"mean", "max", "min", "sum", "median"}
FILL_METHODS = {"none", "linear", "seasonal_slot"}


@dataclass(frozen=True)
class FrameAggregationResult:
    """内存聚合的计算结果：聚合后帧与清洗统计（补缺/重复时间戳计数）。"""

    frame: pd.DataFrame
    source_rows: int
    inserted_timestamp_count: int
    filled_value_count: int
    duplicate_timestamp_count: int


def validate_aggregation_options(method: str, fill_method: str, fill_weeks: int) -> None:
    """聚合参数集中校验：方法白名单与填充窗口正数约束。"""
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"aggregation_method must be one of {sorted(AGGREGATION_METHODS)}")
    if fill_method not in FILL_METHODS:
        raise ValueError(f"aggregation_fill_method must be one of {sorted(FILL_METHODS)}")
    if fill_weeks <= 0:
        raise ValueError("aggregation_fill_weeks must be > 0")


def _seasonal_slot_fill(series: pd.Series, weeks: int) -> pd.Series:
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


def aggregate_frame(
    frame: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    source_freq: str,
    target_freq: str,
    method: str = "mean",
    fill_method: str = "none",
    fill_weeks: int = 4,
) -> FrameAggregationResult:
    """规则化源频率、填充缺口后聚合，返回新表及清洗统计。

    linear/seasonal_slot 使用双向观测，只适用于离线数据准备。
    原始目标中的缺失/非数值仍显式拒绝；填充只处理规则化产生的缺口。
    """
    validate_aggregation_options(method, fill_method, fill_weeks)
    missing_columns = [column for column in (time_col, target_col) if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Aggregation columns not found: {missing_columns}")
    source_rows = len(frame)
    frame = frame.loc[:, [time_col, target_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    if frame[target_col].isna().any():
        raise ValueError(f"Aggregation target '{target_col}' contains non-numeric or missing values")

    series = frame.sort_values(time_col).set_index(time_col).loc[:, target_col].resample(source_freq).mean()
    inserted_count = int(series.isna().sum())
    before_fill = inserted_count
    if fill_method == "linear":
        series = series.interpolate(method="time", limit_direction="both")
    elif fill_method == "seasonal_slot":
        series = _seasonal_slot_fill(series, fill_weeks)

    remaining = int(series.isna().sum())
    if remaining:
        raise ValueError(
            f"Aggregation has {remaining} missing source-frequency values after fill_method={fill_method!r}"
        )
    filled_count = before_fill - remaining
    aggregated = getattr(series.resample(target_freq), method)().reset_index(name=target_col)
    if aggregated[target_col].isna().any():
        raise ValueError("Aggregation produced missing output values")

    return FrameAggregationResult(
        frame=aggregated,
        source_rows=source_rows,
        inserted_timestamp_count=inserted_count,
        filled_value_count=filled_count,
        duplicate_timestamp_count=int(frame[time_col].duplicated().sum()),
    )
