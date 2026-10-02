"""区间列名协议与水平解析：models 与 forecasting 共享的唯一实现。

列名规则（yhat_lower[_label]/yhat_upper[_label]）归本模块，模型层、
策略层、回测与监控一律引用，不各自硬编码列名。组件化区间方法
（IntervalSpec/resolve_interval_plan/conformal 校准）归
forecasting.intervals，其协议函数经 re-export 保持旧引用路径。
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd


def resolve_interval_levels(levels: Sequence[float] | None, alpha: float) -> list[float]:
    """入口级 level 解析：显式 levels 优先，回退 [1 - alpha]；并做范围校验。"""
    if levels is not None and len(levels) > 0:
        resolved = [float(level) for level in levels]
    else:
        resolved = [1.0 - float(alpha)]
    if not resolved:
        raise ValueError("interval levels must be non-empty")
    for level in resolved:
        if not 0 < level < 1:
            raise ValueError(f"interval levels must be in (0, 1), got {level}")
    return resolved


def interval_bound_columns(level: float, multi: bool = False) -> tuple[str, str]:
    """置信水平 → (lower, upper) 输出列名。

    单水平保持 legacy 列名 yhat_lower/yhat_upper；多水平时带百分数后缀
    （80.0 → "80"，80.5 → "80.5"），与 statsforecast 的 lo-80/hi-80 约定对齐。
    """
    label = f"{level * 100:g}"
    if multi:
        return f"yhat_lower_{label}", f"yhat_upper_{label}"
    return "yhat_lower", "yhat_upper"


def iter_bound_pairs(frame: pd.DataFrame):
    """按水平配对迭代区间列：yhat_lower[_suffix] ↔ yhat_upper[_suffix]。"""
    for col in frame.columns:
        if col.startswith("yhat_lower"):
            suffix = col[len("yhat_lower"):]
            upper = f"yhat_upper{suffix}"
            if upper in frame.columns:
                yield col, upper
