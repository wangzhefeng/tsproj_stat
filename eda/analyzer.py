from __future__ import annotations

import pandas as pd

from data_provider.data_preparation import prepare_standard_frame


def prepare_series(df: pd.DataFrame, time_col: str = "ds", target_col: str = "y", freq: str = "D") -> pd.Series:
    """
    将主线清洗后的 DataFrame 转换为 EDA 使用的等频 Series。

    EDA 复用 prepare_standard_frame，随后按 freq 补齐时间索引并插值，
    保证诊断函数拿到的是连续时间序列。
    """
    local_df = prepare_standard_frame(df, time_col=time_col, target_col=target_col, freq=freq)
    
    series = local_df.set_index(time_col)[target_col]
    series = series.asfreq(freq)
    series = series.interpolate(limit_direction="both").dropna()

    if len(series) < 10:
        raise ValueError("series is too short for reliable EDA (need >= 10 samples)")
    
    return series.rename(target_col)
