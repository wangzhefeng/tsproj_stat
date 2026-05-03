from __future__ import annotations

import numpy as np
import pandas as pd

from utils.log_util import logger


def prepare_standard_frame(df: pd.DataFrame, time_col: str = "ds", target_col: str = "y", freq: str = "D") -> pd.DataFrame:
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
    
    # feature filter
    local_df = local_df[[time_col, target_col]].sort_values(time_col).reset_index(drop=True)
    logger.info(f"After feature filter, df:\n {local_df.head()}")
    logger.info(f"After feature filter, df shape: {local_df.shape}")
    
    # interpolate
    local_df[target_col] = local_df[target_col].interpolate(method="linear", limit_direction="both")
    logger.info(f"After interpolate, df:\n {local_df.head()}")
    logger.info(f"After interpolate, df shape: {local_df.shape}")
    
    # drop na
    local_df = local_df.dropna(subset=[target_col]).reset_index(drop=True)
    logger.info(f"After drop na, df:\n {local_df.head()}")
    logger.info(f"After drop na, df shape: {local_df.shape}")

    return local_df
