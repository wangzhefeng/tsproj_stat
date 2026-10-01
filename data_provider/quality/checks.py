"""只读质量检查：缺失率门禁在修复前执行，不推测清洗操作。"""
from __future__ import annotations

import pandas as pd
from .reports import DataQualityReport
from utils.log_util import logger


def check_data_quality(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    max_missing_ratio: float = 0.3,
    validate_freq: bool = True,
    raw_df: pd.DataFrame | None = None,
    freq: str | None = None,
) -> DataQualityReport:
    """
    检查建模前数据质量。

    缺失率超过 max_missing_ratio 直接失败；重复时间戳和频率不规则先记录告警，
    交给调用方决定是否继续。返回值会写入 data_quality.json 作为运行证据。
    """
    if not 0 <= max_missing_ratio <= 1:
        raise ValueError("max_missing_ratio must be in [0, 1]")
    total = len(df)
    raw_rows = len(raw_df) if raw_df is not None else total
    missing = int(df[target_col].isna().sum())
    missing_ratio = missing / total if total > 0 else 0.0
    missing_timestamp_count = 0
    if freq and time_col in df.columns and total > 1:
        times = pd.DatetimeIndex(df[time_col]).sort_values().unique()
        expected = pd.date_range(times[0], times[-1], freq=freq)
        missing_timestamp_count = len(expected.difference(times))

    if missing_ratio > max_missing_ratio:
        raise ValueError(
            f"[DataQuality] target '{target_col}' missing ratio {missing_ratio:.1%} "
            f"exceeds threshold {max_missing_ratio:.1%} ({missing}/{total} rows)"
        )

    dupes = int(df[time_col].duplicated().sum()) if time_col in df.columns else 0
    if dupes > 0:
        logger.warning(f"[DataQuality] {dupes} duplicate timestamps in '{time_col}'")

    freq_irregular = False
    if validate_freq and time_col in df.columns and total > 1:
        times = pd.DatetimeIndex(df[time_col])
        if freq:
            freq_irregular = not times.equals(pd.date_range(times[0], periods=total, freq=freq))
        elif total > 2:
            gaps = df[time_col].diff().dropna()
            freq_irregular = float(gaps.std() / gaps.mean()) > 0.1
        if freq_irregular:
            logger.warning("[DataQuality] timestamps are not on the expected regular grid")

    target_vals = df[target_col].dropna()
    report = DataQualityReport(
        total_rows=total,
        missing_rows=missing,
        missing_ratio=missing_ratio,
        duplicate_timestamps=dupes,
        freq_irregular=freq_irregular,
        target_mean=float(target_vals.mean()) if len(target_vals) > 0 else float("nan"),
        target_std=float(target_vals.std()) if len(target_vals) > 1 else float("nan"),
        time_range=(
            str(df[time_col].iloc[0]) if time_col in df.columns and total > 0 else "",
            str(df[time_col].iloc[-1]) if time_col in df.columns and total > 0 else "",
        ),
        raw_rows=raw_rows,
        clean_rows=total,
        interpolated_value_count=0,
        inserted_timestamp_count=0,
        dropped_row_count=0,
        missing_timestamp_count=missing_timestamp_count,
    )
    logger.info(f"[DataQuality] {report}")
    return report
