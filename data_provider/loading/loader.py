"""数据接入：统一来源读取、结构规范化及修复前质量检查。"""
from __future__ import annotations
import pandas as pd
from .readers import read_frame
from ..cleaning.normalization import normalize_history_frame, normalize_future_frame
from ..quality.checks import check_data_quality
from ..quality.reports import DataQualityReport


class DataLoader:
    """历史数据与未来外生数据加载器。

    只读取、规范化和检查质量；不修复缺失、不切片、不做目标变换或特征工程。
    """

    def __init__(
        self,
        data_path: str | None,
        time_col: str = "ds",
        target_col: str = "y",
        freq: str = "D",
        value_cols: list[str] | None = None,
        future_exog_path: str | None = None,
        future_exog_time_col: str | None = None,
        max_missing_ratio: float = 0.3,
        validate_freq: bool = True,
        data_frame: pd.DataFrame | None = None,
        future_exog_frame: pd.DataFrame | None = None,
    ):
        if data_frame is not None and data_path is not None:
            raise ValueError("data_frame and data_path are mutually exclusive")
        if future_exog_frame is not None and future_exog_path is not None:
            raise ValueError("future_exog_frame and future_exog_path are mutually exclusive")
        self.data_path = data_path
        self.time_col = time_col
        self.target_col = target_col
        self.freq = freq
        self.value_cols = value_cols
        self.future_exog_path = future_exog_path
        self.future_exog_time_col = future_exog_time_col
        self.max_missing_ratio = max_missing_ratio
        self.validate_freq = validate_freq
        # 内存帧直通：面板批量等调用方直接传入已切片的 DataFrame，
        # 与文件路径共用同一条清洗/质检链，避免「每组写 CSV 再读回」的文件中转。
        self.data_frame = data_frame
        self.future_exog_frame = future_exog_frame
        self.quality_report: DataQualityReport | None = None

    def load_data(self) -> pd.DataFrame:
        raw = read_frame(self.data_path, self.data_frame, time_col=self.time_col,
                         target_col=self.target_col, freq=self.freq, allow_demo=True)
        frame = normalize_history_frame(raw, self.time_col, self.target_col, self.freq, self.value_cols)
        self.quality_report = check_data_quality(
            frame, self.target_col, self.time_col, self.max_missing_ratio,
            self.validate_freq, raw_df=raw, freq=self.freq,
        )
        return frame

    def load_future_exog(self, future_exog_cols: list[str], issue_time_col: str | None = None) -> pd.DataFrame | None:
        """读取全部已知未来外生输入；原点对齐和 horizon 选择由 pipeline 负责。"""
        if self.future_exog_path is None and self.future_exog_frame is None:
            return None
        if not future_exog_cols:
            raise ValueError("future_exog_cols must be provided when future exog inputs are set")
        if self.future_exog_time_col is None:
            raise ValueError("future_exog_time_col must be provided when future exog inputs are set")
        raw = read_frame(self.future_exog_path, self.future_exog_frame)
        result = normalize_future_frame(raw, self.future_exog_time_col, future_exog_cols)
        if issue_time_col is not None:
            if issue_time_col == self.future_exog_time_col or issue_time_col in future_exog_cols:
                raise ValueError("issue timestamp must be distinct from time/value columns")
            ordered = raw.copy()
            ordered[self.future_exog_time_col] = pd.to_datetime(ordered[self.future_exog_time_col])
            ordered = ordered.sort_values(self.future_exog_time_col, kind="stable")
            result[issue_time_col] = pd.to_datetime(ordered[issue_time_col], errors="raise").to_numpy()
        return result
