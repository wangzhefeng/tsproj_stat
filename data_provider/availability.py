"""时点可用的外生版本选择；内存表进出，不读取文件或依赖应用配置。"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from data_provider.quality.checks import require_finite


@dataclass(frozen=True)
class FutureExogSource:
    frame: pd.DataFrame
    time_col: str
    columns: tuple[str, ...]
    issue_time_col: str

    def at(self, origin: pd.Timestamp, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """每个有效时刻选 issued_at <= origin 的最新版本；不足不补、不退到实测。"""
        if self.time_col == self.issue_time_col or set(self.columns) & {self.time_col, self.issue_time_col}:
            raise ValueError("forecast time, issue time and value columns must be distinct")
        frame = self.frame.copy()
        for col in (self.time_col, self.issue_time_col):
            frame[col] = pd.to_datetime(frame[col], errors="raise")
            if frame[col].isna().any():
                raise ValueError("forecast timestamps must not be missing")
        if frame.duplicated([self.time_col, self.issue_time_col]).any():
            raise ValueError("duplicate forecast valid/issue timestamp pairs")
        available = frame.loc[frame[self.issue_time_col] <= origin]
        latest = available.sort_values(self.issue_time_col).drop_duplicates(self.time_col, keep="last")
        latest = latest.set_index(self.time_col)
        if not timestamps.isin(latest.index).all():
            raise ValueError("no available as-of forecast covers requested timestamps")
        result = latest.loc[timestamps, list(self.columns)].astype(float).reset_index(drop=True)
        require_finite(result, "as-of future exogenous data")
        return result
