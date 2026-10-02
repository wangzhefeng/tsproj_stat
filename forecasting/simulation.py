"""样本路径模拟（P9）：误差驱动路径集成与分位带。

校准循环与 conformal 区间共用 forecasting.origins.rolling_error_pool
（带符号误差，每窗独立预处理）；bootstrap 抽整窗误差向量，其他分布逐步抽样，
叠加到点预测上得到路径 ensemble。误差分布支持 bootstrap/normal/t/laplace
（t/laplace 逐 step 从误差池 scipy MLE 拟合，拟合失败显式 RAISE）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from models.contracts.inputs import combine_history_frame, to_univariate_series
from models.contracts.validation import validate_horizon
from forecasting.origins import forecast_at_origin, rolling_error_pool
from features.model_inputs import ModelFeatureSpec


SIMULATE_DISTRIBUTIONS = {"bootstrap", "normal", "t", "laplace"}


@dataclass(frozen=True)
class SimulateResult:
    """路径模拟结果：点预测、逐路径长表与可选分位带。

    paths_df 长表三列 path_id/step/value；quantile_df 每分位一列（q10 式）。
    """

    point: pd.Series
    paths_df: pd.DataFrame
    quantile_df: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)


def _fit_step_distribution(errors: np.ndarray, distribution: str) -> dict[int, dict[str, float]]:
    """逐 step 从误差池拟合 t/laplace 参数（MLE，scipy）。

    借鉴 statsforecast simulation 的守卫：拟合样本 <10 拒绝；
    t 的 df<=2（方差无界）拒绝。参数用 loc/scale 保留，逐 step 独立拟合。
    """
    from scipy import stats as scipy_stats

    n_windows = errors.shape[0]
    if n_windows < 10:
        raise ValueError(
            f"distribution={distribution!r} requires >= 10 calibration windows for MLE, got {n_windows}"
        )
    fitted: dict[int, dict[str, float]] = {}
    for step in range(errors.shape[1]):
        sample = errors[:, step]
        if distribution == "t":
            df_est, loc_est, scale_est = scipy_stats.t.fit(sample)
            if df_est <= 2:
                raise ValueError(
                    f"fitted t df={df_est:.2f} <= 2 at step {step + 1}; variance unbounded"
                )
            fitted[step] = {"df": float(df_est), "loc": float(loc_est), "scale": float(scale_est)}
        else:
            loc_est, scale_est = scipy_stats.laplace.fit(sample)
            if scale_est <= 0:
                raise ValueError(f"fitted laplace scale={scale_est:.4g} <= 0 at step {step + 1}")
            fitted[step] = {"loc": float(loc_est), "scale": float(scale_est)}
    return fitted


def _sample_paths(
    errors: np.ndarray,
    distribution: str,
    n_paths: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """从误差池生成 (n_paths, horizon) 的抽样矩阵。

    bootstrap=有放回抽整窗误差行向量（保留步长间同窗相关结构）；
    normal=逐步均值/标准差高斯；t/laplace=逐 step MLE 拟合后按分布抽样。
    """
    if distribution == "bootstrap":
        # 有放回抽整窗误差行向量，保留步长间同窗相关结构；
        # 行索引必须逐路径采样（(n_paths, horizon) 花式索引会广播成 3D）。
        window_idx = rng.integers(0, errors.shape[0], size=n_paths)
        return errors[window_idx]              # (n_paths, horizon)
    if distribution == "normal":
        step_mean = errors.mean(axis=0)
        step_std = errors.std(axis=0, ddof=1) if errors.shape[0] > 1 else np.zeros(horizon)
        return rng.normal(step_mean, step_std, size=(n_paths, horizon))
    from scipy import stats as scipy_stats
    fitted = _fit_step_distribution(errors, distribution)
    draws = np.empty((n_paths, horizon))
    for step in range(horizon):
        params = fitted[step]
        if distribution == "t":
            draws[:, step] = scipy_stats.t.rvs(df=params["df"], loc=params["loc"],
                                               scale=params["scale"], size=n_paths, random_state=rng)
        else:
            draws[:, step] = scipy_stats.laplace.rvs(loc=params["loc"], scale=params["scale"],
                                                     size=n_paths, random_state=rng)
    return draws


def simulate_frame(
    model_builder: Callable,
    history: pd.Series | pd.DataFrame,
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
    feature_spec: ModelFeatureSpec | None = None,
    history_time: pd.Series | None = None,
    future_time: pd.Series | None = None,
) -> SimulateResult:
    """误差驱动样本路径模拟：任意模型 × 任意策略通用。

    与 conformal 同源的滚动起点校准：互不重叠验证段的逐步带符号误差
    构成误差池（n_windows × horizon，原始尺度、每窗独立预处理）；
    bootstrap 抽整窗向量保留步间相关，其他分布逐步抽样，叠加到点预测上。
    t/laplace 需 >=10 个校准窗且拟合参数有效，否则显式失败。
    时间相关与分布漂移下不承诺无条件的路径分布保证；校准样本不足显式失败。
    """
    from utils.random_seed import set_seed

    validate_horizon(horizon)
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be an integer >= 2")
    if error_distribution not in SIMULATE_DISTRIBUTIONS:
        raise ValueError(f"error_distribution must be one of {sorted(SIMULATE_DISTRIBUTIONS)}, got {error_distribution!r}")
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

    errors = rolling_error_pool(
        model_builder, y, x,
        horizon=horizon, strategy=forecast_strategy,
        future_columns=future_columns, processor_builder=processor_builder,
        first_origin=first_origin,
        feature_spec=feature_spec, history_time=history_time,
    )
    if not np.isfinite(errors).all():
        raise ValueError("simulation calibration errors must be finite")

    point_df = forecast_at_origin(
        model_builder, y, h=horizon, strategy=forecast_strategy,
        X_hist=x, X_future=X_future,
        processor_builder=processor_builder, native_intervals=False,
        feature_spec=feature_spec, history_time=history_time, future_time=future_time,
    )
    point = point_df["yhat"].reset_index(drop=True)

    rng = np.random.default_rng(seed if seed is not None else 2026)
    draws = _sample_paths(errors, error_distribution, n_paths, horizon, rng)
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
