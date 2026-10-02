"""评估指标：点误差（mae/rmse/mape/smape/r2/bias/max_error/mase/rmsse）与区间指标（coverage/width/winkler）。

POINT_METRICS 是点指标的单一事实来源：回测窗口指标/汇总、AutoSelector 选优白名单与
方向、多模型对比表的列与排序方向全部由它派生；新增指标只需在此注册。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


# ── 缩放无关指标（跨序列/面板比较用）─────────────────────────────────────────────

def _train_scale(y_train, m: int, squared: bool) -> float:
    """训练窗缩放基准：m 阶差分的绝对值/平方均值；基准不可用（长度不足、非有限、为 0）返回 NaN。"""
    y_train = np.asarray(y_train, dtype=float)
    if y_train.size <= m:
        return float("nan")
    diffs = y_train[m:] - y_train[:-m]
    scale = float(np.mean(diffs ** 2 if squared else np.abs(diffs)))
    return scale if np.isfinite(scale) and scale > 0 else float("nan")


def mase(y_true, y_pred, y_train, m: int = 1):
    """平均绝对缩放误差：mae / 训练窗 m 阶差分绝对值均值（默认 m=1 naive 基准）。

    尺度无关，用于跨序列比较；缩放基准不可用（常数训练窗等）时返回 NaN。
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    scale = _train_scale(y_train, m, squared=False)
    if not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def rmsse(y_true, y_pred, y_train, m: int = 1):
    """均方根缩放误差：sqrt(mse / 训练窗 m 阶差分平方均值)；基准不可用时返回 NaN。"""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    scale = _train_scale(y_train, m, squared=True)
    if not np.isfinite(scale):
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))


# ── 点指标注册表（单一事实来源）────────────────────────────────────────────────

@dataclass(frozen=True)
class PointMetricSpec:
    """点指标注册项。

    func 签名为 (y_true, y_pred)；requires_train=True 的指标追加第三个参数 y_train
    （原始尺度训练窗，用于缩放基准）。higher_is_better 声明选优方向，
    absolute_for_selection 仅转换排名值，不修改原指标的报告值。
    """
    func: Callable[..., float]
    higher_is_better: bool = False
    requires_train: bool = False
    absolute_for_selection: bool = False


POINT_METRICS: dict[str, PointMetricSpec] = {
    "mae": PointMetricSpec(mae),
    "rmse": PointMetricSpec(rmse),
    "mape": PointMetricSpec(mape),
    "smape": PointMetricSpec(smape),
    "mse": PointMetricSpec(mse),
    "r2": PointMetricSpec(r2, higher_is_better=True),
    "bias": PointMetricSpec(bias, absolute_for_selection=True),
    "max_error": PointMetricSpec(max_error),
    "mase": PointMetricSpec(mase, requires_train=True),
    "rmsse": PointMetricSpec(rmsse, requires_train=True),
}


def point_metric_higher_is_better(metric: str) -> bool:
    """指标选优方向；未知指标默认越小越好（兼容 interval_* 等表外指标）。"""
    spec = POINT_METRICS.get(metric)
    return spec.higher_is_better if spec is not None else False


def selection_value(metric: str, value: float | None) -> float:
    """排名值：缺失/非有限统一 NaN；bias 取绝对值，原报告不改。"""
    if value is None or not np.isfinite(value):
        return float("nan")
    spec = POINT_METRICS.get(metric)
    return abs(value) if spec is not None and spec.absolute_for_selection else value


def train_scales(y_train, m: int = 1) -> tuple[float, float]:
    """返回 (mase 基准, rmsse 基准) = (m 阶差分绝对值均值, m 阶差分平方均值)；不可用项为 NaN。

    供回测按 horizon_step 聚合缩放指标时逐窗复用，避免重复扫描训练窗。
    """
    return _train_scale(y_train, m, squared=False), _train_scale(y_train, m, squared=True)


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
