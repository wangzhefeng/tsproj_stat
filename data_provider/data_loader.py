from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from .data_preparation import prepare_standard_frame
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
            demo_series = prepare_standard_frame(
                demo_series,
                time_col=self.time_col,
                target_col=self.target_col,
                freq=self.freq,
            )
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
        df = prepare_standard_frame(
            df,
            time_col=self.time_col,
            target_col=self.target_col,
            freq=self.freq,
        )
        logger.info(f"After prepare_standard_frame, df:\n {df.head()}")
        logger.info(f"After prepare_standard_frame, df shape: {df.shape}")

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
        logger.info(f"history shape: {history.shape}")
        logger.info(f"future shape: {future.shape}")
        
        return history, future
