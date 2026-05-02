from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.demo_data import load_demo_series


@dataclass
class DataLoader:

    data_path: str | None
    time_col: str = "ds"
    target_col: str = "y"
    freq: str = "D"

    def load_data(self) -> pd.DataFrame:
        if self.data_path is None:
            return load_demo_series(time_col=self.time_col, target_col=self.target_col, freq=self.freq)

        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(path)
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in data columns {list(df.columns)}")
        df = self._normalize_time_col(df)
        return self._apply_missing_value_policy(df)

    def split_history_future(self, df: pd.DataFrame, history_size: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(df) < history_size + horizon:
            raise ValueError("Not enough samples for requested history_size + horizon")
        
        history = df.iloc[-(history_size + horizon):-horizon].reset_index(drop=True)
        future = df.iloc[-horizon:].reset_index(drop=True)
        
        return history, future

    def _normalize_time_col(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.time_col not in out.columns:
            # Compatibility fallback for old CSVs that only contain the target column.
            out[self.time_col] = pd.date_range("2000-01-01", periods=len(out), freq=self.freq)
        else:
            out[self.time_col] = pd.to_datetime(out[self.time_col])
        return out

    def _apply_missing_value_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[[self.time_col, self.target_col]]
            .dropna(subset=[self.target_col])
            .sort_values(self.time_col)
            .reset_index(drop=True)
        )
