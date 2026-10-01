"""表格结构规范化：不插值、不补时间轴、不删除缺失行。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_frame(
    frame: pd.DataFrame, time_col: str, value_cols: list[str],
    *, freq: str = "D", allow_generated_time: bool = False,
) -> pd.DataFrame:
    """复制并统一时间、列和数值类型；保留缺失观测供窗口处理。"""
    local = frame.copy()
    if time_col not in local:
        if not allow_generated_time:
            raise ValueError(f"time column '{time_col}' not found")
        local[time_col] = pd.date_range("2000-01-01", periods=len(local), freq=freq)
    local[time_col] = pd.to_datetime(local[time_col], errors="raise")
    if local[time_col].isna().any():
        raise ValueError("time column contains missing timestamps")
    columns = list(dict.fromkeys(value_cols))
    missing = [col for col in columns if col not in local]
    if missing:
        raise ValueError(f"value columns not found: {missing}")
    local = local[[time_col, *columns]].sort_values(time_col, kind="stable").reset_index(drop=True)
    for col in columns:
        local[col] = pd.to_numeric(local[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return local


def normalize_history_frame(
    frame: pd.DataFrame, time_col: str = "ds", target_col: str = "y",
    freq: str = "D", value_cols: list[str] | None = None,
) -> pd.DataFrame:
    if target_col not in frame:
        raise ValueError(f"target_col '{target_col}' not found in data columns {list(frame.columns)}")
    return normalize_frame(frame, time_col, [target_col, *(value_cols or [])],
                           freq=freq, allow_generated_time=True)


def normalize_future_frame(frame: pd.DataFrame, time_col: str, value_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in value_cols if col not in frame]
    if missing:
        raise ValueError(f"future_exog_cols missing from future exog data: {missing}")
    return normalize_frame(frame, time_col, value_cols)
