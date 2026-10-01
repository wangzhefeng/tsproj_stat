"""数据来源读取；不清洗、不选择建模窗口。"""
from pathlib import Path
import pandas as pd
from utils.demo_data import load_demo_series


def read_frame(path: str | None, frame: pd.DataFrame | None, *,
               time_col: str = "ds", target_col: str = "y", freq: str = "D",
               allow_demo: bool = False) -> pd.DataFrame:
    if path is not None and frame is not None:
        raise ValueError("data_frame and data_path are mutually exclusive")
    if frame is not None:
        return frame.copy(deep=True)
    if path is None:
        if allow_demo:
            return load_demo_series(time_col=time_col, target_col=target_col, freq=freq)
        raise ValueError("data source is required")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)
