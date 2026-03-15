from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_timeseries(data_path: str | None, time_col: str = "ds", target_col: str = "y", freq: str = "D") -> pd.DataFrame:
    if data_path is None:
        x = np.arange(120)
        y = 10 + 0.2 * x + np.sin(x / 5)
        return pd.DataFrame({time_col: pd.date_range("2024-01-01", periods=len(x), freq=freq), target_col: y})

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in csv columns: {list(df.columns)}")
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
    else:
        df[time_col] = pd.date_range("2000-01-01", periods=len(df), freq=freq)
    return df[[time_col, target_col]].copy()
