from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class TimeSeriesDataset:
    df: pd.DataFrame
    time_col: str = "ds"
    target_col: str = "y"


def train_test_split_by_time(df: pd.DataFrame, test_size: int):
    """按时间顺序切分数据，保留时序结构。"""
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be in (0, len(df))")
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test
