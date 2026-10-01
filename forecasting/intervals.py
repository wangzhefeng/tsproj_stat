"""区间组件：IntervalSpec 挂载 + 策略合法性裁决 + 原始尺度误差校准。

区间方法（none/native/conformal）作为可挂载组件：合法组合经
resolve_interval_plan 返回执行计划；非法组合（如 native × recursive/dirrec）
在进入推理前显式 RAISE 并给出可行替代，不再产出 NaN 区间列。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from data_provider.cleaning.imputation import repair_history_frame, require_finite
from models.contracts.inputs import combine_history_frame, to_univariate_series
from models.contracts.validation import validate_horizon
from forecasting.strategies import run_point_inference, run_interval_inference


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

    def resolved_levels(self) -> tuple[float, ...]:
        """有效置信水平：显式 levels 优先，否则回退 [1 - alpha]。"""
        if self.levels is not None and len(self.levels) > 0:
            return tuple(self.levels)
        return (1.0 - self.alpha,)


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
    """按水平配对迭代区间列：yhat_lower[_suffix] ↔ yhat_upper[_suffix]。

    列名协议的唯一归属是本模块（interval_bound_columns）；回测与策略层
    一律经本函数配对，不各自硬编码列名规则。
    """
    for col in frame.columns:
        if col.startswith("yhat_lower"):
            suffix = col[len("yhat_lower"):]
            upper = f"yhat_upper{suffix}"
            if upper in frame.columns:
                yield col, upper


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


def _forecast_origin(builder, y, h, strategy, X_hist, X_future, processor_builder, native_intervals, alpha, levels=None):
    frame = combine_history_frame(y, X_hist)
    frame, audit = repair_history_frame(frame, list(frame.columns))
    y = frame.iloc[:, 0]
    if X_future is not None:
        require_finite(X_future, "future exogenous data")
    proc = processor_builder() if processor_builder is not None else None
    values = proc.fit_transform(y) if proc is not None and proc.enabled else y
    frame.iloc[:, 0] = values.to_numpy()
    if native_intervals:
        result = run_interval_inference(builder, values, h, strategy, frame, X_future, alpha, levels)
    else:
        result = run_point_inference(builder, values, h, strategy, frame, X_future).to_frame("yhat")
    if proc is not None and proc.enabled:
        # 区间边界与点预测必须一起逆变换；乘法变换可能交换上下界。
        lower_cols = [c for c in result.columns if c.startswith("yhat_lower")]
        upper_cols = [c for c in result.columns if c.startswith("yhat_upper")]
        for column in result.columns:
            result[column] = proc.inverse_forecast(result[column]).to_numpy()
        for lower_col, upper_col in zip(lower_cols, upper_cols):
            bounds = result[[lower_col, upper_col]].to_numpy()
            result[lower_col] = np.minimum(bounds[:, 0], bounds[:, 1])
            result[upper_col] = np.maximum(bounds[:, 0], bounds[:, 1])
    if not np.isfinite(result["yhat"]).all():
        raise ValueError("non-finite point prediction in calibration/forecast")
    result.attrs["history_filled_value_count"] = audit.filled_value_count
    result.attrs["history_repair_policy"] = audit.policy
    return result


def predict_frame(model_builder, history, horizon, forecast_strategy, X_hist=None, X_future=None,
                  processor_builder=None, interval_method="none", alpha=0.05, n_windows=20,
                  levels: Sequence[float] | None = None,
                  spec: IntervalSpec | None = None):
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
        result = _forecast_origin(model_builder, y, horizon, forecast_strategy, x, X_future,
                                  processor_builder, interval_method == "native",
                                  alpha, resolved_levels)
        result.attrs["interval_method"] = interval_method
        result.attrs["interval_levels"] = resolved_levels
        return result
    if isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 2:
        raise ValueError("calibration n_windows must be an integer >= 2")
    # 逐水平有限样本 rank；任一水平不可达即显式失败（不静默丢弃该水平）。
    ranks = {level: math.ceil((n_windows + 1) * level) for level in resolved_levels}
    for level, rank in ranks.items():
        if rank > n_windows:
            raise ValueError(
                f"calibration windows insufficient for finite-sample interval level {level:g}"
            )
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
    sorted_errors = np.sort(errors, axis=0)
    result = _forecast_origin(model_builder, y, horizon, forecast_strategy, x, X_future,
                              processor_builder, False, alpha)
    for level in resolved_levels:
        lower_col, upper_col = interval_bound_columns(level, multi=multi)
        radius = sorted_errors[ranks[level] - 1]
        result[lower_col] = result.yhat - radius
        result[upper_col] = result.yhat + radius
    result.attrs.update(interval_method="conformal", calibration_windows=n_windows,
                        calibration_first_origin=first_origin,
                        calibration_ranks={level: ranks[level] for level in resolved_levels},
                        interval_levels=resolved_levels, interval_alpha=alpha)
    return result


# ##############################
# 样本路径模拟（P9）
# ##############################

@dataclass(frozen=True)
class SimulateResult:
    """路径模拟结果：点预测、逐路径长表与可选分位带。

    paths_df 长表三列 path_id/step/value；quantile_df 每分位一列（q10 式）。
    """

    point: pd.Series
    paths_df: pd.DataFrame
    quantile_df: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


def simulate_frame(
    model_builder,
    history,
    horizon: int,
    forecast_strategy: str,
    n_paths: int = 100,
    error_distribution: str = "bootstrap",
    n_windows: int = 20,
    quantiles: Sequence[float] | None = None,
    seed: int | None = None,
    X_hist=None,
    X_future=None,
    processor_builder=None,
) -> SimulateResult:
    """误差驱动样本路径模拟：任意模型 × 任意策略通用。

    与 conformal 同源的滚动起点校准：互不重叠验证段的逐步带符号误差
    构成误差池（n_windows × horizon，原始尺度、每窗独立预处理）；
    每条路径对每个步长独立抽取误差（bootstrap=有放回抽整窗行向量，
    normal=逐步均值/标准差高斯），叠加到点预测上得到路径 ensemble。
    时间相关与分布漂移下不承诺无条件的路径分布保证；校准样本不足显式失败。
    """
    from utils.random_seed import set_seed

    validate_horizon(horizon)
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be an integer >= 2")
    if error_distribution not in {"bootstrap", "normal"}:
        raise ValueError(f"error_distribution must be bootstrap or normal, got {error_distribution!r}")
    resolved_quantiles: list[float] = list(quantiles) if quantiles else []
    for q in resolved_quantiles:
        if not 0 < q < 1:
            raise ValueError(f"quantiles must be in (0, 1), got {q}")
    if isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 2:
        raise ValueError("n_windows must be an integer >= 2")

    set_seed(seed if seed is not None else 2026)

    y = to_univariate_series(history).astype(float)
    x = combine_history_frame(history, X_hist)
    future_columns = [] if X_future is None else list(X_future.columns)
    if any(c not in x for c in future_columns):
        raise ValueError("simulation requires historical values of future exogenous columns")
    first_origin = len(y) - n_windows * horizon
    if first_origin < 3:
        raise ValueError(
            f"simulation history insufficient: need n_windows * horizon + 3 rows "
            f"({n_windows} * {horizon} + 3)"
        )
    # 与 conformal 相同的校准循环（带符号，非绝对值）。
    scores: list[np.ndarray] = []
    for origin in range(first_origin, len(y), horizon):
        future = x.iloc[origin:origin + horizon][future_columns].reset_index(drop=True) if future_columns else None
        pred = _forecast_origin(model_builder, y.iloc[:origin], horizon, forecast_strategy,
                                x.iloc[:origin], future, processor_builder, False, 0.05)
        scores.append(y.iloc[origin:origin + horizon].to_numpy() - pred.yhat.to_numpy())
    errors = np.asarray(scores)
    if not np.isfinite(errors).all():
        raise ValueError("simulation calibration errors must be finite")

    point_df = _forecast_origin(model_builder, y, horizon, forecast_strategy,
                                x, X_future, processor_builder, False, 0.05)
    point = point_df["yhat"].reset_index(drop=True)

    rng = np.random.default_rng(seed if seed is not None else 2026)
    if error_distribution == "bootstrap":
        # 有放回抽整窗误差行向量，保留步长间同窗相关结构；
        # 行索引必须逐路径采样（(n_paths, horizon) 花式索引会广播成 3D）。
        window_idx = rng.integers(0, n_windows, size=n_paths)
        draws = errors[window_idx]                  # (n_paths, horizon)
    else:
        step_mean = errors.mean(axis=0)
        step_std = errors.std(axis=0, ddof=1) if n_windows > 1 else np.zeros(horizon)
        draws = rng.normal(step_mean, step_std, size=(n_paths, horizon))
    paths = point.to_numpy()[None, :] + draws       # (n_paths, horizon)

    rows = []
    for path_id in range(n_paths):
        for step in range(horizon):
            rows.append({"path_id": path_id + 1, "step": step + 1, "value": float(paths[path_id, step])})
    paths_df = pd.DataFrame(rows)

    quantile_df = None
    if resolved_quantiles:
        data = {"q" + f"{q * 100:g}": np.quantile(paths, q, axis=0) for q in resolved_quantiles}
        quantile_df = pd.DataFrame(data)

    metadata = {
        "n_paths": int(n_paths),
        "error_distribution": error_distribution,
        "n_windows": int(n_windows),
        "quantiles": resolved_quantiles,
        "seed": int(seed if seed is not None else 2026),
    }
    return SimulateResult(point=point, paths_df=paths_df, quantile_df=quantile_df, metadata=metadata)
