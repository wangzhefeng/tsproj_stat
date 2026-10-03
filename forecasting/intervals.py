"""区间组件：IntervalSpec 挂载 + 策略合法性裁决 + 原始尺度误差校准。

区间方法（none/native/conformal）作为可挂载组件：合法组合经
resolve_interval_plan 返回执行计划；非法组合（如 native × recursive/dirrec）在
进入推理前显式 RAISE 并给出可行替代，不再产出 NaN 区间列。
列名协议（interval_bound_columns/iter_bound_pairs/resolve_interval_levels）
唯一实现位于 models.contracts.intervals，本模块 re-export 保持旧路径。
单一原点编排与滚动误差池归 forecasting.origins；样本路径模拟归
forecasting.simulation。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from models.contracts.inputs import combine_history_frame, to_univariate_series
from models.contracts.intervals import (
    interval_bound_columns,
    iter_bound_pairs,
    resolve_interval_levels,
)
from models.contracts.validation import validate_horizon
from forecasting.origins import forecast_at_origin, rolling_error_pool

# 区间列名协议的唯一实现位于 models.contracts.intervals；本模块 re-export
# 保持 forecasting.intervals.* 的旧引用路径（能力位组件仍在本模块）。
__all__ = [
    "IntervalSpec",
    "IntervalPlan",
    "resolve_interval_plan",
    "predict_frame",
    "resolve_interval_levels",
    "interval_bound_columns",
    "iter_bound_pairs",
]


@dataclass(frozen=True)
class IntervalSpec:
    """区间方法组件的完整参数（strategy 由调用方单独提供）。

    levels 是小数置信水平列表（如 (0.80, 0.95)）；未显式设置时回退
    [1 - alpha]，保持单水平旧行为。多水平下输出列带水平后缀。
    """

    method: str = "none"            # none | native | conformal
    alpha: float = 0.05
    conformal_n_windows: int = 20
    levels: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.method not in {"none", "native", "conformal"}:
            raise ValueError(f"interval method must be none, native or conformal, got {self.method!r}")
        if not 0 < self.alpha < 1:
            raise ValueError("interval alpha must be in (0, 1)")
        if self.method == "conformal" and self.conformal_n_windows < 2:
            raise ValueError("conformal_n_windows must be >= 2")
        # 空列表与 None 同义（未设置 → 回退 [1 - alpha]），与 config 层一致；
        # 规范化进属性，使 spec.levels 恒为有效非空元组。
        resolved = self.levels if self.levels else (1.0 - self.alpha,)
        for level in resolved:
            if not 0 < level < 1:
                raise ValueError(f"interval levels must be in (0, 1), got {level}")
        object.__setattr__(self, "levels", tuple(resolved))


@dataclass(frozen=True)
class IntervalPlan:
    """resolve_interval_plan 的裁决结果：组合是否合法与拒绝理由。"""

    method: str
    allowed: bool
    reason: str | None = None


def resolve_interval_plan(spec: IntervalSpec, strategy: str) -> IntervalPlan:
    """裁决「区间方法 × 预测策略」组合的合法性。

    - none：任意策略合法；
    - native：模型原生解析区间只在一次拟合语义下有效
      （native/single_step/direct），recursive/dirrec 的逐步重拟合
      会破坏区间的解析连续性，显式拒绝；
    - conformal：任意策略合法（校准以点预测误差为基础）。
    """
    if spec.method == "none":
        return IntervalPlan(method="none", allowed=True)
    if spec.method == "native":
        if strategy in {"recursive", "dirrec"}:
            return IntervalPlan(
                method="native",
                allowed=False,
                reason=(
                    f"native intervals are unavailable under strategy={strategy!r} "
                    "(stepwise refit breaks analytic continuity); "
                    "use interval_method=conformal instead"
                ),
            )
        return IntervalPlan(method="native", allowed=True)
    return IntervalPlan(method="conformal", allowed=True)


def _conformal_ranks(resolved_levels: list[float], n_windows: int) -> dict[float, int]:
    """逐水平有限样本 rank：ceil((n+1)·level)，> n_windows 即不可达。"""
    return {level: math.ceil((n_windows + 1) * level) for level in resolved_levels}


def _conformal_radius(
    errors: np.ndarray,
    resolved_levels: list[float],
    n_windows: int,
) -> dict[float, np.ndarray]:
    """逐水平有限样本 rank → 校准半径（绝对误差顺序统计量）。

    任一水平有限样本不可达即显式失败（不静默丢弃该水平）。
    """
    ranks = _conformal_ranks(resolved_levels, n_windows)
    for level, rank in ranks.items():
        if rank > n_windows:
            raise ValueError(
                f"calibration windows insufficient for finite-sample interval level {level:g}"
            )
    sorted_errors = np.sort(np.abs(errors), axis=0)
    return {level: sorted_errors[ranks[level] - 1] for level in resolved_levels}


def predict_frame(model_builder, history, horizon, forecast_strategy, X_hist=None, X_future=None,
                  processor_builder=None, interval_method="none", alpha=0.05, n_windows=20,
                  levels: Sequence[float] | None = None,
                  spec: IntervalSpec | None = None, history_time=None,
                  exog_future_known=False, future_source=None):
    """输入原始 history；返回原始尺度点预测与可选区间。

    spec 显式传入时以其为准（interval_method/alpha/n_windows/levels 兼容参数仍支持）；
    入口先经 resolve_interval_plan 裁决，非法组合直接 RAISE。
    conformal 使用互不重叠验证段的逐步绝对误差顺序统计量；多水平共享同一次
    校准循环（拟合成本不随水平数增加），按水平取各自 rank。
    时间相关和分布漂移下不承诺无条件覆盖保证；样本不足显式失败。
    """
    validate_horizon(horizon)
    if spec is not None:
        interval_method, alpha, n_windows = spec.method, spec.alpha, spec.conformal_n_windows
        if levels is None:
            levels = spec.levels
    if interval_method not in {"none", "native", "conformal"}:
        raise ValueError("interval_method must be none, native or conformal")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    resolved_levels = resolve_interval_levels(levels, alpha)
    plan = resolve_interval_plan(
        IntervalSpec(method=interval_method, alpha=alpha, conformal_n_windows=n_windows,
                     levels=tuple(resolved_levels)),
        forecast_strategy,
    )
    if not plan.allowed:
        raise ValueError(plan.reason or "unsupported interval x strategy combination")
    y = to_univariate_series(history).astype(float)
    x = combine_history_frame(history, X_hist)
    multi = len(resolved_levels) > 1
    if interval_method != "conformal":
        result = forecast_at_origin(model_builder, y, h=horizon, strategy=forecast_strategy,
                                    X_hist=x, X_future=X_future, processor_builder=processor_builder,
                                    native_intervals=interval_method == "native",
                                    alpha=alpha, levels=resolved_levels)
        result.attrs["interval_method"] = interval_method
        result.attrs["interval_levels"] = resolved_levels
        return result
    if isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 2:
        raise ValueError("calibration n_windows must be an integer >= 2")
    first_origin = len(y) - n_windows * horizon
    if first_origin < 3:
        raise ValueError("calibration history insufficient: need n_windows * horizon + 3 rows")
    future_columns = [] if X_future is None else list(X_future.columns)
    if any(c not in x for c in future_columns):
        raise ValueError("calibration requires historical values of future exogenous columns")
    errors = rolling_error_pool(
        model_builder, y, x,
        horizon=horizon, strategy=forecast_strategy,
        future_columns=future_columns, processor_builder=processor_builder,
        first_origin=first_origin,
        history_time=history_time, exog_future_known=exog_future_known, future_source=future_source,
    )
    if not np.isfinite(errors).all():
        raise ValueError("calibration errors must be finite")
    radii = _conformal_radius(errors, resolved_levels, n_windows)
    result = forecast_at_origin(model_builder, y, h=horizon, strategy=forecast_strategy,
                                X_hist=x, X_future=X_future, processor_builder=processor_builder,
                                native_intervals=False, alpha=alpha)
    for level in resolved_levels:
        lower_col, upper_col = interval_bound_columns(level, multi=multi)
        radius = radii[level]
        result[lower_col] = result.yhat - radius
        result[upper_col] = result.yhat + radius
    result.attrs.update(interval_method="conformal", calibration_windows=n_windows,
                        calibration_exog_policy=("as_of_forecast" if future_source is not None else "known_in_advance") if future_columns else "none",
                        calibration_first_origin=first_origin,
                        calibration_ranks=_conformal_ranks(resolved_levels, n_windows),
                        interval_levels=resolved_levels, interval_alpha=alpha)
    return result
