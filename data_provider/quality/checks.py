"""只读质量检查：缺失率门禁在修复前执行，不推测清洗操作。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .reports import DataQualityReport
from utils.log_util import logger


def require_finite(frame: pd.DataFrame | pd.Series, role: str) -> None:
    """未来输入和评分真值不得通过插值伪造；空输入不是有效观测。"""
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{role} contains missing or non-finite observations (or is empty)")


def require_regular_time(times: pd.Series | pd.DatetimeIndex, freq: str | None = None,
                         role: str = "history") -> None:
    """只读门禁：有效、唯一、递增、等频；不排序或补齐输入。"""
    index = pd.DatetimeIndex(pd.to_datetime(times, errors="raise"))
    if index.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{role} timestamps must be non-empty, valid, unique and increasing")
    resolved = freq or (pd.infer_freq(index) if len(index) >= 3 else None)
    if resolved is None:
        if len(index) >= 3:
            raise ValueError(f"{role} timestamps must be regular; supply an explicit frequency")
        return  # 没有期望频率时，至多两个点只能检查顺序和唯一性。
    if not index.equals(pd.date_range(index[0], periods=len(index), freq=resolved)):
        raise ValueError(f"{role} timestamps are not regular at frequency {resolved!r}")


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

    columns = [col for col in df.columns if col != time_col]
    raw_values = raw_df if raw_df is not None else df
    raw_missing = {}
    coercion_failed = {}
    nonfinite = {}
    for col in columns:
        raw_col = raw_values[col]
        numeric = pd.to_numeric(raw_col, errors="coerce")
        raw_missing[col] = int(raw_col.isna().sum())
        coercion_failed[col] = int((raw_col.notna() & numeric.isna()).sum())
        nonfinite[col] = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
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
        missing_by_column={col: int(df[col].isna().sum()) for col in columns},
        raw_missing_by_column=raw_missing,
        coercion_failed_by_column=coercion_failed,
        nonfinite_by_column=nonfinite,
    )
    logger.info(f"[DataQuality] {report}")
    return report
