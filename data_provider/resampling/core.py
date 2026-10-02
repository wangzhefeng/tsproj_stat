"""内存时间序列聚合：不读写文件、不依赖应用配置。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries import offsets

from data_provider.cleaning.seasonal import seasonal_slot_fill
from data_provider.quality.checks import require_finite

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


def _require_downsampling(source_freq: str, target_freq: str) -> None:
    """固定步长要求整倍数；日历桶仅接受完整日的细分或同频。

    不用月/季度的近似纳秒长度推断方向；未定义的跨日历组合明确拒绝。
    """
    source, target = to_offset(source_freq), to_offset(target_freq)
    if source is None or target is None or source.n <= 0 or target.n <= 0:
        raise ValueError("aggregation frequencies must be positive")
    if source == target:
        return
    fixed = (offsets.Tick, offsets.Day)  # pandas 3 的 Day 不再继承 Tick。
    if isinstance(source, fixed) and isinstance(target, fixed):
        if target.nanos >= source.nanos and target.nanos % source.nanos == 0:
            return
    calendar = (offsets.Week, offsets.MonthBegin, offsets.MonthEnd,
                offsets.QuarterBegin, offsets.QuarterEnd, offsets.YearBegin, offsets.YearEnd)
    if isinstance(source, fixed) and isinstance(target, calendar):
        if pd.Timedelta(days=1).value % source.nanos == 0:
            return
    raise ValueError(f"aggregation requires verified downsampling or identical frequencies: {source_freq} -> {target_freq}")


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
    _require_downsampling(source_freq, target_freq)
    if frame.empty:
        raise ValueError("Aggregation input must not be empty")
    missing_columns = [column for column in (time_col, target_col) if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Aggregation columns not found: {missing_columns}")
    source_rows = len(frame)
    frame = frame.loc[:, [time_col, target_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    if frame[time_col].isna().any():
        raise ValueError("Aggregation timestamps must be valid and non-missing")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    if frame[target_col].isna().any():
        raise ValueError(f"Aggregation target '{target_col}' contains non-numeric or missing values")
    require_finite(frame[target_col], "Aggregation target")

    series = frame.sort_values(time_col).set_index(time_col).loc[:, target_col].resample(source_freq).mean()
    if not pd.DatetimeIndex(frame[time_col]).isin(series.index).all():
        raise ValueError("Aggregation timestamps must align with the source grid")
    inserted_count = int(series.isna().sum())
    before_fill = inserted_count
    if fill_method == "linear":
        series = series.interpolate(method="time", limit_direction="both")
    elif fill_method == "seasonal_slot":
        series = seasonal_slot_fill(series, fill_weeks)

    remaining = int(series.isna().sum())
    if remaining:
        raise ValueError(
            f"Aggregation has {remaining} missing source-frequency values after fill_method={fill_method!r}"
        )
    filled_count = before_fill - remaining
    aggregated = getattr(series.resample(target_freq), method)().reset_index(name=target_col)
    require_finite(aggregated[target_col], "Aggregation output")

    return FrameAggregationResult(
        frame=aggregated,
        source_rows=source_rows,
        inserted_timestamp_count=inserted_count,
        filled_value_count=filled_count,
        duplicate_timestamp_count=int(frame[time_col].duplicated().sum()),
    )
