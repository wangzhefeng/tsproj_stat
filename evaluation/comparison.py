"""多模型对比表组装与按指标选优：model_comparison.csv 与 auto_select 的纯计算部分。

对比列与选优方向的唯一事实来源是 evaluation.metrics.POINT_METRICS；
本模块内存进内存出，落盘仍由 pipeline.runner 收口。
"""
from __future__ import annotations

import pandas as pd

from evaluation.metrics import POINT_METRICS, point_metric_higher_is_better, selection_value

# 对比表在点指标之外附带的窗口统计列。
COMPARISON_EXTRA_KEYS = ("window_count", "failed_windows", "survivor_bias")


def build_comparison_frame(test_summaries: dict[str, dict], metric: str) -> pd.DataFrame:
    """把各模型回测汇总载荷组装成对比表，并按 metric 的注册表方向排序。

    行键 = 注册表点指标 + 窗口统计列；载荷中缺失的键跳过（容忍旧产物无新指标列）。
    """
    keys = (*POINT_METRICS.keys(), *COMPARISON_EXTRA_KEYS)
    rows: list[dict] = []
    for name, payload in test_summaries.items():
        row = {"model_name": name}
        for key in keys:
            if key in payload:
                row[key] = payload[key]
        rows.append(row)
    df = pd.DataFrame(rows)
    if metric in df.columns:
        df = df.sort_values(
            by=metric, ascending=not point_metric_higher_is_better(metric),
            key=lambda values: values.map(lambda v: selection_value(metric, v)), kind="stable",
        ).reset_index(drop=True)
    return df


def select_best_model(test_summaries: dict[str, dict], metric: str) -> str:
    """按注册表方向及偏差语义选优；缺失/非有限跳过，全部不可用即 RAISE。"""
    higher = point_metric_higher_is_better(metric)
    best_name: str | None = None
    best_value: float | None = None
    for name, payload in test_summaries.items():
        if metric not in payload:
            continue
        value = selection_value(metric, payload[metric])
        if value != value:  # NaN guard
            continue
        if best_value is None or (value > best_value if higher else value < best_value):
            best_name, best_value = name, value
    if best_name is None:
        raise RuntimeError(f"no readable {metric} scores for auto_select")
    return best_name
