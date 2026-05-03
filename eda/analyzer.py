from __future__ import annotations

import pandas as pd

from data_provider.data_preparation import prepare_standard_frame


def prepare_series(df: pd.DataFrame, time_col: str = "ds", target_col: str = "y", freq: str = "D") -> pd.Series:
    """
    timeseries prepare
    """
    # timeseries prepare
    local_df = prepare_standard_frame(df, time_col=time_col, target_col=target_col, freq=freq)
    
    series = local_df.set_index(time_col)[target_col]
    series = series.asfreq(freq)
    series = series.interpolate(limit_direction="both").dropna()

    if len(series) < 10:
        raise ValueError("series is too short for reliable EDA (need >= 10 samples)")
    
    return series.rename(target_col)
