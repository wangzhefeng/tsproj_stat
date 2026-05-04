from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from .data_preparation import prepare_future_exog_frame, prepare_standard_frame
from utils.demo_data import load_demo_series
from utils.log_util import logger


@dataclass
class DataQualityReport:
    """清洗后数据质量摘要，用于日志与 data_quality.json。"""
    total_rows: int
    missing_rows: int
    missing_ratio: float
    duplicate_timestamps: int
    freq_irregular: bool
    target_mean: float
    target_std: float
    time_range: tuple[str, str]

    def __str__(self) -> str:
        return (
            f"DataQualityReport("
            f"total={self.total_rows}, missing={self.missing_rows}({self.missing_ratio:.1%}), "
            f"dup_ts={self.duplicate_timestamps}, freq_irregular={self.freq_irregular}, "
            f"target_mean={self.target_mean:.3f}, target_std={self.target_std:.3f}, "
            f"range=[{self.time_range[0]}, {self.time_range[1]}])"
        )

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "missing_rows": self.missing_rows,
            "missing_ratio": round(self.missing_ratio, 6),
            "duplicate_timestamps": self.duplicate_timestamps,
            "freq_irregular": self.freq_irregular,
            "target_mean": round(self.target_mean, 6),
            "target_std": round(self.target_std, 6),
            "time_range_start": self.time_range[0],
            "time_range_end": self.time_range[1],
        }


def check_data_quality(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    max_missing_ratio: float = 0.3,
    validate_freq: bool = True,
) -> DataQualityReport:
    """
    检查建模前数据质量。

    缺失率超过 max_missing_ratio 直接失败；重复时间戳和频率不规则先记录告警，
    交给调用方决定是否继续。返回值会写入 data_quality.json 作为运行证据。
    """
    total = len(df)
    missing = int(df[target_col].isna().sum())
    missing_ratio = missing / total if total > 0 else 0.0

    if missing_ratio > max_missing_ratio:
        raise ValueError(
            f"[DataQuality] target '{target_col}' missing ratio {missing_ratio:.1%} "
            f"exceeds threshold {max_missing_ratio:.1%} ({missing}/{total} rows)"
        )

    dupes = int(df[time_col].duplicated().sum()) if time_col in df.columns else 0
    if dupes > 0:
        logger.warning(f"[DataQuality] {dupes} duplicate timestamps in '{time_col}'")

    freq_irregular = False
    if validate_freq and time_col in df.columns and total > 2:
        try:
            times = pd.to_datetime(df[time_col], errors="coerce").dropna()
            if len(times) > 2:
                gaps = times.diff().dropna()
                cv = gaps.std() / gaps.mean()
                freq_irregular = float(cv) > 0.1
                if freq_irregular:
                    logger.warning(f"[DataQuality] irregular time intervals detected (CV={float(cv):.3f})")
        except Exception:
            pass

    target_vals = df[target_col].dropna()
    report = DataQualityReport(
        total_rows=total,
        missing_rows=missing,
        missing_ratio=missing_ratio,
        duplicate_timestamps=dupes,
        freq_irregular=freq_irregular,
        target_mean=float(target_vals.mean()) if len(target_vals) > 0 else float("nan"),
        target_std=float(target_vals.std()) if len(target_vals) > 1 else float("nan"),
        time_range=(
            str(df[time_col].iloc[0]) if time_col in df.columns and total > 0 else "",
            str(df[time_col].iloc[-1]) if time_col in df.columns and total > 0 else "",
        ),
    )
    logger.info(f"[DataQuality] {report}")
    return report


@dataclass
class DataLoader:
    """历史数据与未来外生数据加载器。

    该类只负责读取、标准清洗和质量检查；不承担预处理、特征工程或模型切分逻辑。
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
    ):
        self.data_path = data_path
        self.time_col = time_col
        self.target_col = target_col
        self.freq = freq
        self.value_cols = value_cols
        self.future_exog_path = future_exog_path
        self.future_exog_time_col = future_exog_time_col
        self.max_missing_ratio = max_missing_ratio
        self.validate_freq = validate_freq
        self.quality_report: DataQualityReport | None = None

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
                value_cols=self.value_cols,
            )
            logger.info(f"Loaded demo series dataset:\n {demo_series.head()}")
            logger.info(f"Loaded demo series dataset shape: {demo_series.shape}")
            self.quality_report = check_data_quality(
                demo_series, self.target_col, self.time_col,
                max_missing_ratio=self.max_missing_ratio,
                validate_freq=self.validate_freq,
            )
            return demo_series
        # ------------------------------
        # 使用本地数据
        # ------------------------------
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded raw data:\n {df.head()}")
        logger.info(f"Loaded raw data shape: {df.shape}")
        df = prepare_standard_frame(
            df,
            time_col=self.time_col,
            target_col=self.target_col,
            freq=self.freq,
            value_cols=self.value_cols,
        )
        logger.info(f"After prepare_standard_frame, df:\n {df.head()}")
        logger.info(f"After prepare_standard_frame, df shape: {df.shape}")
        self.quality_report = check_data_quality(
            df, self.target_col, self.time_col,
            max_missing_ratio=self.max_missing_ratio,
            validate_freq=self.validate_freq,
        )
        return df

    def load_future_exog(self, future_exog_cols: list[str], horizon: int) -> pd.DataFrame | None:
        """读取预测期外生变量，并截取到预测 horizon 长度。"""
        if self.future_exog_path is None:
            return None
        if not future_exog_cols:
            raise ValueError("future_exog_cols must be provided when future_exog_path is set")
        if self.future_exog_time_col is None:
            raise ValueError("future_exog_time_col must be provided when future_exog_path is set")

        path = Path(self.future_exog_path)
        if not path.exists():
            raise FileNotFoundError(f"Future exog file not found: {self.future_exog_path}")

        df = pd.read_csv(path)
        prepared = prepare_future_exog_frame(
            df,
            time_col=self.future_exog_time_col,
            value_cols=future_exog_cols,
        )
        if len(prepared) < horizon:
            raise ValueError("Future exogenous data has fewer rows than requested horizon")
        return prepared.iloc[:horizon].reset_index(drop=True)
    
    def split_history_future(self, df: pd.DataFrame, history_size: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        从序列尾部切出训练历史窗口和未来评估窗口。

        Args:
            df: 已完成标准清洗和可选预处理的数据。
            history_size: 预测阶段使用的历史窗口长度。
            horizon: 预测未来步数，也是尾部保留的未来窗口长度。

        Raises:
            ValueError: 数据长度不足以同时覆盖 history_size 与 horizon。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: history_df 与 future_df，均已重置索引。
        """
        if len(df) < history_size + horizon:
            raise ValueError("Not enough samples for requested history_size + horizon")
        
        history = df.iloc[-(history_size + horizon):-horizon].reset_index(drop=True)
        future = df.iloc[-horizon:].reset_index(drop=True)
        logger.info(f"history shape: {history.shape}")
        logger.info(f"future shape: {future.shape}")
        
        return history, future
