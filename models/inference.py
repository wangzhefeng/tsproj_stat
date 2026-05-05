from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pandas as pd

from data_provider.data_transfer import combine_history_frame, to_dataframe, to_univariate_series


FORECAST_STRATEGIES = {"single_step", "direct", "recursive", "dirrec"}
WINDOW_MODES = {"expanding", "sliding"}


def normalize_forecast_strategy(
    forecast_strategy: str | None,
) -> str:
    """标准化多步预测策略名称。"""
    candidate = str(forecast_strategy or "direct").strip().lower()
    if candidate not in FORECAST_STRATEGIES:
        raise ValueError(f"forecast_strategy must be one of {sorted(FORECAST_STRATEGIES)}")
    return candidate


def normalize_window_mode(window_mode: str | None) -> str:
    """标准化 rolling backtest 的窗口模式。"""
    candidate = (window_mode or "expanding").strip().lower()
    if candidate not in WINDOW_MODES:
        raise ValueError(f"backtest_window_mode must be one of {sorted(WINDOW_MODES)}")
    return candidate


def validate_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")


def validate_single_step_horizon(strategy: str, horizon: int) -> None:
    if strategy == "single_step" and horizon != 1:
        raise ValueError("single_step forecast_strategy requires predict_horizon/backtest_horizon == 1")


def _coerce_single_value(value: pd.Series | float | np.floating | int) -> float:
    if isinstance(value, pd.Series):
        if value.empty:
            raise ValueError("predict_one returned empty Series")
        return float(value.iloc[0])
    return float(value)


def _future_prefix(X_future: pd.DataFrame | None, length: int) -> pd.DataFrame | None:
    if X_future is None:
        return None
    return to_dataframe(X_future).iloc[:length].reset_index(drop=True)


def _future_row(X_future: pd.DataFrame | None, index: int) -> pd.DataFrame | None:
    if X_future is None:
        return None
    future = to_dataframe(X_future).reset_index(drop=True)
    if index >= len(future):
        raise ValueError(
            f"future exogenous rows ({len(future)}) are insufficient for inference step {index + 1}"
        )
    return future.iloc[index : index + 1].reset_index(drop=True)


def _append_history_frame(
    history_frame: pd.DataFrame,
    next_target: float,
    next_future: pd.DataFrame | None,
) -> pd.DataFrame:
    next_row = history_frame.iloc[-1].copy()
    next_row.iloc[0] = float(next_target)
    if next_future is not None:
        for col in next_future.columns:
            if col in next_row.index:
                next_row[col] = float(next_future.iloc[0][col])
    return pd.concat([history_frame, pd.DataFrame([next_row])], ignore_index=True)


def _predict_one(model, X_future_one: pd.DataFrame | None = None) -> float:
    return _coerce_single_value(model.predict_one(X_future_one=X_future_one))


def _predict_direct_step(model, step: int, X_future_prefix: pd.DataFrame | None = None) -> float:
    pred = model.predict(step, X_future=X_future_prefix)
    series = pred if isinstance(pred, pd.Series) else pd.Series(pred)
    if len(series) != step:
        raise ValueError(f"predict({step}) returned length {len(series)}")
    return float(series.iloc[-1])


def run_point_inference(
    model_builder: Callable[[], object],
    history: pd.Series | pd.DataFrame,
    horizon: int,
    forecast_strategy: str,
    X_hist: pd.DataFrame | None = None,
    X_future: pd.DataFrame | None = None,
) -> pd.Series:
    """执行点预测的统一多步推理编排。

    single_step: 只允许 horizon=1；
    direct: 每个预测步重新拟合一个模型并取对应步长的最后一个预测值；
    recursive: 每一步把上一轮预测追加回历史，再预测下一步；
    dirrec: 逐步重建模型，同时使用递归扩展后的历史。
    """
    strategy = normalize_forecast_strategy(forecast_strategy)
    validate_horizon(horizon)
    validate_single_step_horizon(strategy, horizon)

    history_series = to_univariate_series(history).astype(float).reset_index(drop=True)
    history_frame = combine_history_frame(history, X_hist).astype(float).reset_index(drop=True)
    future_frame = None if X_future is None else to_dataframe(X_future).astype(float).reset_index(drop=True)

    if strategy == "single_step":
        # 单步策略只拟合一次，严格对应 predict_one 契约。
        model = model_builder()
        first_future = _future_row(future_frame, 0)
        model.fit(history_series, X_hist=history_frame, X_future=first_future)
        return pd.Series([_predict_one(model, first_future)], name="yhat")

    if strategy == "direct":
        # direct 策略按 step=1..horizon 独立拟合，避免把预测值递归写回历史。
        preds: list[float] = []
        for step_idx in range(horizon):
            model = model_builder()
            prefix = _future_prefix(future_frame, step_idx + 1)
            model.fit(history_series, X_hist=history_frame, X_future=prefix)
            preds.append(_predict_direct_step(model, step_idx + 1, prefix))
        return pd.Series(preds, name="yhat")

    hist = history_series.copy()
    hist_frame = history_frame.copy()
    preds = []
    for step_idx in range(horizon):
        # recursive/dirrec 都会把预测追加回历史；dirrec 每步重建模型，recursive 复用模型副本。
        model = model_builder() if strategy == "dirrec" else copy.deepcopy(model_builder())
        next_future = _future_row(future_frame, step_idx)
        model.fit(hist, X_hist=hist_frame, X_future=next_future)
        next_val = _predict_one(model, next_future)
        preds.append(next_val)
        hist = pd.concat([hist, pd.Series([next_val])], ignore_index=True)
        hist_frame = _append_history_frame(hist_frame, next_val, next_future)
    return pd.Series(preds, name="yhat")


def run_interval_inference(
    model_builder: Callable[[], object],
    history: pd.Series | pd.DataFrame,
    horizon: int,
    forecast_strategy: str,
    X_hist: pd.DataFrame | None = None,
    X_future: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """执行区间预测编排。

    只有 single_step/direct 可以从每个直接预测模型取原生区间；
    recursive/dirrec 暂时返回点预测和 NaN 区间，避免伪造不可靠置信区间。
    """
    strategy = normalize_forecast_strategy(forecast_strategy)
    point = run_point_inference(
        model_builder=model_builder,
        history=history,
        horizon=horizon,
        forecast_strategy=strategy,
        X_hist=X_hist,
        X_future=X_future,
    )
    if strategy in {"recursive", "dirrec"}:
        return pd.DataFrame(
            {
                "yhat": point.values,
                "yhat_lower": np.full(len(point), np.nan),
                "yhat_upper": np.full(len(point), np.nan),
            }
        )

    history_series = to_univariate_series(history).astype(float).reset_index(drop=True)
    history_frame = combine_history_frame(history, X_hist).astype(float).reset_index(drop=True)
    future_frame = None if X_future is None else to_dataframe(X_future).astype(float).reset_index(drop=True)

    if strategy == "single_step":
        model = model_builder()
        first_future = _future_row(future_frame, 0)
        model.fit(history_series, X_hist=history_frame, X_future=first_future)
        result = model.predict_with_intervals(1, X_future=first_future, alpha=alpha)
        return result.reset_index(drop=True)

    rows: list[dict[str, float]] = []
    for step_idx in range(horizon):
        model = model_builder()
        prefix = _future_prefix(future_frame, step_idx + 1)
        model.fit(history_series, X_hist=history_frame, X_future=prefix)
        result = model.predict_with_intervals(step_idx + 1, X_future=prefix, alpha=alpha).reset_index(drop=True)
        row = result.iloc[-1]
        rows.append(
            {
                "yhat": float(row["yhat"]),
                "yhat_lower": float(row["yhat_lower"]) if pd.notna(row["yhat_lower"]) else np.nan,
                "yhat_upper": float(row["yhat_upper"]) if pd.notna(row["yhat_upper"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)
