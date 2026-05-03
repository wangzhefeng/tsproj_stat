from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.demo_data import load_demo_series
from utils.log_util import logger


@dataclass
class DataLoader:

    def __init__(self, data_path: str | None, time_col: str = "ds", target_col: str = "y", freq: str = "D"):
        self.data_path = data_path
        self.time_col = time_col
        self.target_col = target_col
        self.freq = freq

    def load_data(self) -> pd.DataFrame:
        # ------------------------------
        # 使用 demo series dataset
        # ------------------------------
        if self.data_path is None:
            demo_series = load_demo_series(time_col=self.time_col, target_col=self.target_col, freq=self.freq)
            logger.info(f"Loaded demo series dataset:\n {demo_series.head()}")
            logger.info(f"Loaded demo series dataset shape: {demo_series.shape}")
            return demo_series 
        # ------------------------------
        # 使用本地数据
        # ------------------------------
        # history data path
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        # read data
        df = pd.read_csv(path)
        logger.info(f"Loaded raw data:\n {df.head()}")
        logger.info(f"Loaded raw data shape: {df.shape}")
        # time col 规范化
        df = self._normalize_time_col(df)
        logger.info(f"After _normalize_time_col, df:\n {df.head()}")
        logger.info(f"After _normalize_time_col, df shape: {df.shape}")
        # 缺失值处理
        df = self._apply_missing_value_policy(df)
        logger.info(f"After _apply_missing_value_policy, df:\n {df.head()}")
        logger.info(f"After _apply_missing_value_policy, df shape: {df.shape}")

        return df
    
    def split_history_future(self, df: pd.DataFrame, history_size: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        split history and future 

        Args:
            df (pd.DataFrame): _description_
            history_size (int): _description_
            horizon (int): _description_

        Raises:
            ValueError: _description_

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: _description_
        """
        if len(df) < history_size + horizon:
            raise ValueError("Not enough samples for requested history_size + horizon")
        
        history = df.iloc[-(history_size + horizon):-horizon].reset_index(drop=True)
        future = df.iloc[-horizon:].reset_index(drop=True)
        
        return history, future

    def _normalize_time_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        time_col 规范化
        """
        out_df = df.copy()
        if self.time_col not in out_df.columns:
            # Compatibility fallback for old CSVs that only contain the target column.
            out_df[self.time_col] = pd.date_range("2000-01-01", periods=len(out_df), freq=self.freq)
        else:
            out_df[self.time_col] = pd.to_datetime(out_df[self.time_col])
        
        return out_df

    def _normalize_target_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        target_col 规范化
        """
        out_df = df.copy()
        if self.target_col not in out_df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in data columns {list(df.columns)}")
        else:
            out_df[self.target_col] = pd.to_numeric(out_df[self.target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out_df = out_df[[self.time_col, self.target_col]]
        
        return out_df

    def _apply_missing_value_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        缺失值处理 
        """
        out_df = df.copy()
        if isinstance(out_df.index, pd.DatetimeIndex):
            series = series.asfreq(self.freq)
            series = series.interpolate(limit_direction="both").dropna()
            series = series.sort_index()
        out_df = (
            df[[self.time_col, self.target_col]]
            .interpolate(method="linear", limit_direction="both")
            .dropna(subset=[self.target_col])
            .sort_values(self.time_col)
            .reset_index(drop=True)
        )

        return out_df
