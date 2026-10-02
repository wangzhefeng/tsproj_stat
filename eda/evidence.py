"""预测导向的 EDA 证据：时间单位、分析视图、分段周期验证与异常明细。

全部为内存计算，不修复输入、不把全量分量传给预测模型。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import detrend


def sample_seconds(freq: str | None) -> float | None:
    if not freq:
        return None
    offset = pd.tseries.frequencies.to_offset(freq)
    if isinstance(offset, pd.offsets.Day):
        return float(offset.n * 86400)
    try:
        return float(offset.nanos / 1e9)
    except ValueError:
        return None  # 月/季度等日历周期不能假定固定时长。


def period_label(points: int | float, freq: str | None) -> str:
    seconds = sample_seconds(freq)
    if seconds is None:
        return f"{points:g} 点（频率 {freq or '未知'}）"
    hours = points * seconds / 3600
    return f"{points:g} 点（{hours:g} 小时）"


def analysis_views(series: pd.Series) -> dict[str, pd.Series]:
    values = series.to_numpy(dtype=float)
    return {
        "raw": series,
        "linear_detrended": pd.Series(detrend(values), index=series.index, name=series.name),
        "difference": series.diff().dropna(),
    }


def period_candidates(period: int, freq: str | None, peaks: list[int]) -> list[int]:
    candidates = [period, *peaks]
    seconds = sample_seconds(freq)
    if seconds:
        for duration in (86400, 7 * 86400):
            points = duration / seconds
            if points >= 2 and points.is_integer():
                candidates.append(int(points))
    return sorted({p for p in candidates if p >= 2})


def validate_periods(series: pd.Series, periods: list[int]) -> list[dict]:
    """线性去趋势后，以两个连续半段互相验证槽位均值剖面。

    每半段至少两周期；双向样本外 R²>=0.1 且剖面相关>=0.5 才标为稳定候选。
    这是探索性筛选（候选使用全样本提出），不等于独立留出集预测成绩。
    """
    values = series.to_numpy(dtype=float)
    slots = np.arange(len(values))
    split = len(values) // 2
    result = []
    for period in periods:
        row: dict = {"period": period, "status": "insufficient", "stable": False,
                     "reason": "each half needs >= 2 periods", "forward_r2": None,
                     "backward_r2": None, "profile_correlation": None}
        if split < 2 * period:
            result.append(row)
            continue
        halves = [detrend(values[:split]), detrend(values[split:])]
        phases = [slots[:split] % period, slots[split:] % period]
        profiles = [np.bincount(phase, weights=y, minlength=period) /
                    np.bincount(phase, minlength=period) for y, phase in zip(halves, phases)]
        variance = [float(np.var(y)) for y in halves]
        tolerance = np.finfo(float).eps * max(1.0, float(np.mean(values ** 2)))
        if min(variance) <= tolerance or min(float(np.var(p)) for p in profiles) <= tolerance:
            row.update(status="insufficient", reason="negligible detrended/profile variance")
        else:
            forward = 1 - float(np.mean((halves[1] - profiles[0][phases[1]]) ** 2)) / variance[1]
            backward = 1 - float(np.mean((halves[0] - profiles[1][phases[0]]) ** 2)) / variance[0]
            corr = float(np.asarray(np.corrcoef(np.asarray(profiles)))[0, 1])
            stable = min(forward, backward) >= 0.1 and corr >= 0.5
            row.update(status="ok", stable=stable, reason="bidirectional half-sample profile validation",
                       forward_r2=forward, backward_r2=backward, profile_correlation=corr)
        result.append(row)
    return result


def outlier_details(series: pd.Series, residual: pd.Series | None) -> pd.DataFrame:
    """保存全局 IQR/Z-score 和 STL 残差 MAD 标记；标记不等于错误值。"""
    values = series.to_numpy(dtype=float)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    rows: list[dict] = []
    methods = [
        ("global_iqr", values, q1 - 1.5 * iqr, q3 + 1.5 * iqr),
        ("global_zscore", values, float(series.mean() - 3 * series.std()),
         float(series.mean() + 3 * series.std())),
    ]
    if residual is not None:
        r = residual.to_numpy(dtype=float)
        center = float(np.median(r))
        # MAD=0 的近确定性序列使用浮点误差下限，而非除零或把所有点当异常。
        scale = max(1.4826 * float(np.median(np.abs(r - center))),
                    np.finfo(float).eps * max(1.0, float(np.max(np.abs(values)))) * 100)
        methods.append(("stl_residual_mad", r, center - 3 * scale, center + 3 * scale))
    for name, tested, lower, upper in methods:
        for position in np.flatnonzero((tested < lower) | (tested > upper)):
            i = int(position)
            rows.append({"time": str(series.index[i]), "value": values[i], "method": name,
                         "tested_value": tested[i], "lower": lower, "upper": upper})
    return pd.DataFrame(rows, columns=["time", "value", "method", "tested_value", "lower", "upper"])
