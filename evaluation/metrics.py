"""评估指标：点误差（mae/rmse/mape/smape/r2/bias/max_error）与区间指标（coverage/width/winkler）。"""
from __future__ import annotations

import numpy as np


def _to_arrays(y_true, y_pred):
    """将指标输入统一转为 float ndarray，避免 pandas index 影响计算。"""
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def mae(y_true, y_pred):
    """平均绝对误差：mean(|y_true - y_pred|)，与目标同量纲，越小越好。"""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    """均方误差：mean((y_true - y_pred)^2)，对大误差更敏感。"""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    """均方根误差：sqrt(mse)，与目标同量纲。"""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8):
    """平均绝对百分比误差（小数形式，非百分数）；|y_true| < eps 时按 eps 兜底防除零。"""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def smape(y_true, y_pred, eps: float = 1e-8):
    """对称 MAPE：2|y_true-y_pred| / (|y_true|+|y_pred|)，分母过小按 eps 兜底。"""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def r2(y_true, y_pred):
    """决定系数：1 - SS_res/SS_tot；样本 <2 或 y_true 无方差时返回 NaN。"""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(total, 0.0):
        return float("nan")
    residual = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - residual / total)


def bias(y_true, y_pred):
    """平均偏差：mean(y_pred - y_true)，正值表示系统性高估。"""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def max_error(y_true, y_pred):
    """最大绝对误差：max(|y_true - y_pred|)，刻画最坏单点。"""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


# ── 区间预测评估指标 ────────────────────────────────────────────────────────────

def coverage(y_true, lower, upper) -> float:
    """实际值落在区间内的比例，目标值为 1 - alpha（如 95% 区间应接近 0.95）。"""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = ~(np.isnan(lower) | np.isnan(upper))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])))


def interval_width(lower, upper) -> float:
    """区间平均宽度，在保证覆盖率的前提下越窄越好。"""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = ~(np.isnan(lower) | np.isnan(upper))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(upper[mask] - lower[mask]))


def winkler_score(y_true, lower, upper, alpha: float = 0.05) -> float:
    """
    Winkler score：综合惩罚区间宽度与区间外点（越小越好）。
    - 在区间内：得分 = 区间宽度
    - 低于下界：得分 = 区间宽度 + (2/alpha)*(下界 - 实际值)
    - 高于上界：得分 = 区间宽度 + (2/alpha)*(实际值 - 上界)
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = ~(np.isnan(lower) | np.isnan(upper))
    if mask.sum() == 0:
        return float("nan")
    yt, lo, hi = y_true[mask], lower[mask], upper[mask]
    width = hi - lo
    penalty = np.where(yt < lo, 2.0 / alpha * (lo - yt),
               np.where(yt > hi, 2.0 / alpha * (yt - hi), 0.0))
    return float(np.mean(width + penalty))
