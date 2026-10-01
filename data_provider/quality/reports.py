"""规范化数据质量报告；操作计数与缺口统计分别命名。"""
from dataclasses import dataclass


@dataclass
class DataQualityReport:
    """规范化后、修复前的数据质量摘要，用于日志与 data_quality.json。"""
    total_rows: int
    missing_rows: int
    missing_ratio: float
    duplicate_timestamps: int
    freq_irregular: bool
    target_mean: float
    target_std: float
    time_range: tuple[str, str]
    raw_rows: int = 0
    clean_rows: int = 0
    interpolated_value_count: int = 0
    inserted_timestamp_count: int = 0
    dropped_row_count: int = 0
    missing_timestamp_count: int = 0

    def __str__(self) -> str:
        """一行式摘要，供日志直接打印。"""
        return (
            f"DataQualityReport("
            f"total={self.total_rows}, missing={self.missing_rows}({self.missing_ratio:.1%}), "
            f"dup_ts={self.duplicate_timestamps}, freq_irregular={self.freq_irregular}, "
            f"target_mean={self.target_mean:.3f}, target_std={self.target_std:.3f}, "
            f"range=[{self.time_range[0]}, {self.time_range[1]}])"
        )

    def to_dict(self) -> dict:
        """序列化为 data_quality.json 的扁平字典（浮点保留 6 位）。"""
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
            "raw_rows": self.raw_rows,
            "clean_rows": self.clean_rows,
            "interpolated_value_count": self.interpolated_value_count,
            "inserted_timestamp_count": self.inserted_timestamp_count,
            "dropped_row_count": self.dropped_row_count,
            "missing_timestamp_count": self.missing_timestamp_count,
        }
