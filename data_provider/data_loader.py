from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DataLoader:
    data_path: str | None
    time_col: str = "ds"
    target_col: str = "y"
    freq: str = "D"

    def load_data(self) -> pd.DataFrame:
        if self.data_path is None:
            x = np.arange(200)
            y = 10 + 0.15 * x + np.sin(x / 8)
            return pd.DataFrame({
                self.time_col: pd.date_range("2024-01-01", periods=len(x), freq=self.freq),
                self.target_col: y,
            })

        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(path)
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in data columns {list(df.columns)}")

        if self.time_col not in df.columns:
            df[self.time_col] = pd.date_range("2000-01-01", periods=len(df), freq=self.freq)
        else:
            df[self.time_col] = pd.to_datetime(df[self.time_col])

        df = df[[self.time_col, self.target_col]].dropna(subset=[self.target_col]).sort_values(self.time_col).reset_index(drop=True)
        return df

    def split_history_future(self, df: pd.DataFrame, history_size: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(df) < history_size + horizon:
            raise ValueError("Not enough samples for requested history_size + horizon")
        history = df.iloc[-(history_size + horizon):-horizon].reset_index(drop=True)
        future = df.iloc[-horizon:].reset_index(drop=True)
        return history, future
