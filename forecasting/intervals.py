"""区间组件：IntervalSpec 挂载 + 策略合法性裁决 + 原始尺度误差校准。

区间方法（none/native/conformal）作为可挂载组件：合法组合经
resolve_interval_plan 返回执行计划；非法组合（如 native × recursive/dirrec）
在进入推理前显式 RAISE 并给出可行替代，不再产出 NaN 区间列。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_provider.data_transfer import combine_history_frame, to_univariate_series
from forecasting.strategies import run_point_inference, run_interval_inference, validate_horizon


@dataclass(frozen=True)
class IntervalSpec:
    """区间方法组件的完整参数（strategy 由调用方单独提供）。"""

    method: str = "none"            # none | native | conformal
    alpha: float = 0.05
    conformal_n_windows: int = 20

    def __post_init__(self) -> None:
        if self.method not in {"none", "native", "conformal"}:
            raise ValueError(f"interval method must be none, native or conformal, got {self.method!r}")
        if not 0 < self.alpha < 1:
            raise ValueError("interval alpha must be in (0, 1)")
        if self.method == "conformal" and self.conformal_n_windows < 2:
            raise ValueError("conformal_n_windows must be >= 2")


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


def _forecast_origin(builder, y, h, strategy, X_hist, X_future, processor_builder, native_intervals, alpha):
    frame = combine_history_frame(y, X_hist)
    proc = processor_builder() if processor_builder is not None else None
    values = proc.fit_transform(y) if proc is not None and proc.enabled else y
    frame.iloc[:, 0] = values.to_numpy()
    if native_intervals:
        result = run_interval_inference(builder, values, h, strategy, frame, X_future, alpha)
    else:
        result = run_point_inference(builder, values, h, strategy, frame, X_future).to_frame("yhat")
    if proc is not None and proc.enabled:
        # 区间边界与点预测必须一起逆变换；乘法变换可能交换上下界。
        for column in result.columns:
            result[column] = proc.inverse_forecast(result[column]).to_numpy()
        if "yhat_lower" in result:
            bounds = result[["yhat_lower", "yhat_upper"]].to_numpy()
            result["yhat_lower"] = np.minimum(bounds[:, 0], bounds[:, 1])
            result["yhat_upper"] = np.maximum(bounds[:, 0], bounds[:, 1])
    if not np.isfinite(result["yhat"]).all():
        raise ValueError("non-finite point prediction in calibration/forecast")
    return result


def predict_frame(model_builder, history, horizon, forecast_strategy, X_hist=None, X_future=None,
                  processor_builder=None, interval_method="none", alpha=0.05, n_windows=20,
                  spec: IntervalSpec | None = None):
    """输入原始 history；返回原始尺度点预测与可选区间。

    spec 显式传入时以其为准（interval_method/alpha/n_windows 兼容参数仍支持）；
    入口先经 resolve_interval_plan 裁决，非法组合直接 RAISE。
    conformal 使用互不重叠验证段的逐步绝对误差顺序统计量。
    时间相关和分布漂移下不承诺无条件覆盖保证；样本不足显式失败。
    """
    validate_horizon(horizon)
    if spec is not None:
        interval_method, alpha, n_windows = spec.method, spec.alpha, spec.conformal_n_windows
    if interval_method not in {"none", "native", "conformal"}:
        raise ValueError("interval_method must be none, native or conformal")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    plan = resolve_interval_plan(IntervalSpec(method=interval_method, alpha=alpha, conformal_n_windows=n_windows), forecast_strategy)
    if not plan.allowed:
        raise ValueError(plan.reason or "unsupported interval x strategy combination")
    y = to_univariate_series(history).astype(float)
    x = combine_history_frame(history, X_hist)
    if interval_method != "conformal":
        result = _forecast_origin(model_builder, y, horizon, forecast_strategy, x, X_future,
                                  processor_builder, interval_method == "native", alpha)
        result.attrs["interval_method"] = interval_method
        return result
    if isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 2:
        raise ValueError("calibration n_windows must be an integer >= 2")
    rank = math.ceil((n_windows + 1) * (1 - alpha))
    if rank > n_windows:
        raise ValueError("calibration windows insufficient for finite-sample interval level")
    first_origin = len(y) - n_windows * horizon
    if first_origin < 3:
        raise ValueError("calibration history insufficient: need n_windows * horizon + 3 rows")
    future_columns = [] if X_future is None else list(X_future.columns)
    if any(c not in x for c in future_columns):
        raise ValueError("calibration requires historical values of future exogenous columns")
    scores = []
    for origin in range(first_origin, len(y), horizon):
        future = x.iloc[origin:origin + horizon][future_columns].reset_index(drop=True) if future_columns else None
        pred = _forecast_origin(model_builder, y.iloc[:origin], horizon, forecast_strategy,
                                x.iloc[:origin], future, processor_builder, False, alpha)
        scores.append(np.abs(y.iloc[origin:origin + horizon].to_numpy() - pred.yhat.to_numpy()))
    errors = np.asarray(scores)
    if not np.isfinite(errors).all():
        raise ValueError("calibration errors must be finite")
    radius = np.sort(errors, axis=0)[rank - 1]
    result = _forecast_origin(model_builder, y, horizon, forecast_strategy, x, X_future,
                              processor_builder, False, alpha)
    result["yhat_lower"] = result.yhat - radius
    result["yhat_upper"] = result.yhat + radius
    result.attrs.update(interval_method="conformal", calibration_windows=n_windows,
                        calibration_first_origin=first_origin, calibration_rank=rank,
                        calibration_radius=radius.tolist(), interval_alpha=alpha)
    return result
