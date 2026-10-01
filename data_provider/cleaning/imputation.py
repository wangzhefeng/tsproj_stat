"""在调用方已切出的历史窗口内修复数值，不接触评估真值。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RepairAudit:
    """执行器直接记录操作数量；窗口外数据不可见。"""
    filled_value_count: int
    filled_by_column: dict[str, int]
    inserted_timestamp_count: int = 0
    dropped_row_count: int = 0
    policy: str = "window_local_linear"


def repair_history_frame(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, RepairAudit]:
    """窗口内双向线性插值；边界用窗口内有效端点，全缺列显式失败。"""
    out = frame.copy()
    counts: dict[str, int] = {}
    for col in dict.fromkeys(columns):
        values = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = values.isna()
        repaired = values.interpolate(method="linear", limit_direction="both")
        if repaired.isna().any() or repaired.empty:
            raise ValueError(f"history window column '{col}' has no usable observations")
        counts[col] = int((missing & repaired.notna()).sum())
        out[col] = repaired
    return out, RepairAudit(sum(counts.values()), counts)


def require_finite(frame: pd.DataFrame | pd.Series, role: str) -> None:
    """未来输入和评分真值不得通过插值伪造。"""
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{role} contains missing or non-finite observations")
