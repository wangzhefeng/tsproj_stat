from __future__ import annotations

import numpy as np
import pandas as pd

from utils.log_util import logger


def prepare_standard_frame(
    df: pd.DataFrame,
    time_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    timeseries prepare
    """
    local_df = df.copy()

    # time_col check
    if time_col not in local_df.columns:
        local_df[time_col] = pd.date_range("2000-01-01", periods=len(local_df), freq=freq)
    else:
        local_df[time_col] = pd.to_datetime(local_df[time_col])
    logger.info(f"After time_col check, df:\n {local_df.head()}")
    logger.info(f"After time_col check, df shape: {local_df.shape}")
    
    # target_col check
    if target_col not in local_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in data columns {list(df.columns)}")
    else:
        local_df[target_col] = pd.to_numeric(local_df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    logger.info(f"After target_col check, df:\n {local_df.head()}")
    logger.info(f"After target_col check, df shape: {local_df.shape}")
    
    selected_value_cols = [target_col]
    if value_cols is not None:
        selected_value_cols = []
        for col in value_cols:
            if col not in local_df.columns:
                raise ValueError(f"value column '{col}' not found in data columns {list(df.columns)}")
            selected_value_cols.append(col)
        if target_col not in selected_value_cols:
            selected_value_cols = [target_col, *selected_value_cols]

    # feature filter
    local_df = local_df[[time_col, *selected_value_cols]].sort_values(time_col).reset_index(drop=True)
    logger.info(f"After feature filter, df:\n {local_df.head()}")
    logger.info(f"After feature filter, df shape: {local_df.shape}")

    for col in selected_value_cols:
        local_df[col] = pd.to_numeric(local_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        local_df[col] = local_df[col].interpolate(method="linear", limit_direction="both")
    logger.info(f"After interpolate, df:\n {local_df.head()}")
    logger.info(f"After interpolate, df shape: {local_df.shape}")

    # drop na
    local_df = local_df.dropna(subset=selected_value_cols).reset_index(drop=True)
    logger.info(f"After drop na, df:\n {local_df.head()}")
    logger.info(f"After drop na, df shape: {local_df.shape}")

    return local_df


def prepare_future_exog_frame(df: pd.DataFrame, time_col: str, value_cols: list[str]) -> pd.DataFrame:
    local_df = df.copy()
    if time_col not in local_df.columns:
        raise ValueError(f"future time column '{time_col}' not found in data columns {list(df.columns)}")
    missing = [col for col in value_cols if col not in local_df.columns]
    if missing:
        raise ValueError(f"future_exog_cols missing from future exog data: {missing}")

    local_df[time_col] = pd.to_datetime(local_df[time_col])
    local_df = local_df[[time_col, *value_cols]].sort_values(time_col).reset_index(drop=True)
    for col in value_cols:
        local_df[col] = pd.to_numeric(local_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        local_df[col] = local_df[col].interpolate(method="linear", limit_direction="both")
    return local_df.dropna(subset=value_cols).reset_index(drop=True)
