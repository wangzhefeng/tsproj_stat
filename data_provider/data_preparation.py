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
    标准化历史时间序列数据。

    这是 DataLoader 与 EDA 共用的清洗入口：统一处理时间列、目标列、
    多源 value_cols、排序、数值化、inf/NaN 和线性插值。
    """
    local_df = df.copy()

    # 时间列缺失时生成规则时间索引，支持 demo/smoke 数据快速运行。
    if time_col not in local_df.columns:
        local_df[time_col] = pd.date_range("2000-01-01", periods=len(local_df), freq=freq)
    else:
        local_df[time_col] = pd.to_datetime(local_df[time_col])
    logger.info(f"After time_col check, df:\n {local_df.head()}")
    logger.info(f"After time_col check, df shape: {local_df.shape}")
    
    # 目标列是单目标 yhat 契约的核心字段，缺失时必须直接失败。
    if target_col not in local_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in data columns {list(df.columns)}")
    else:
        local_df[target_col] = pd.to_numeric(local_df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    logger.info(f"After target_col check, df:\n {local_df.head()}")
    logger.info(f"After target_col check, df shape: {local_df.shape}")
    
    selected_value_cols = [target_col]
    if value_cols is not None:
        # 多源输入场景下保留配置指定的协变量列；历史数据仍必须保留 target_col。
        selected_value_cols = []
        for col in value_cols:
            if col not in local_df.columns:
                raise ValueError(f"value column '{col}' not found in data columns {list(df.columns)}")
            selected_value_cols.append(col)
        if target_col not in selected_value_cols:
            selected_value_cols = [target_col, *selected_value_cols]

    # 只保留主线需要的 canonical 列，避免无关原始字段进入模型输入。
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
    """标准化独立未来外生变量文件。

    未来外生数据只负责提供预测期已知变量，不要求包含 target_col。
    """
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
