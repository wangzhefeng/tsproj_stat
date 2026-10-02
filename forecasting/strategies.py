"""多步预测策略与推理编排：native/single_step/direct/recursive/dirrec。

策略语义：native 一次拟合原生多步；direct 逐步重拟合并取对应步长末值
（兼容语义，非 horizon-specific 监督学习）；recursive 逐步把预测写回历史；
dirrec 逐步重建模型且用递归扩展后的历史。模型能力门禁由 checked_model_builder
在推理前统一校验。
"""
from __future__ import annotations

import copy
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from models.contracts.inputs import combine_history_frame, to_dataframe, to_univariate_series
from models.contracts.validation import validate_horizon
from models.base import BaseStatModel
from features.model_inputs import FutureFeatures
from config.strategy import (  # noqa: F401  — 兼容旧导入路径的 re-export
    FORECAST_STRATEGIES,
    WINDOW_MODES,
    normalize_forecast_strategy,
    normalize_window_mode,
    validate_single_step_horizon,
)


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


def _validate_update_path(strategy: str, model_builder, X_future) -> None:
    """forward 快速路径前置门禁：策略、模型能力、未来外生三重校验。"""
    if strategy != "recursive":
        raise ValueError("use_update requires forecast_strategy=recursive")
    if X_future is not None and to_dataframe(X_future).shape[1] > 0:
        raise ValueError("use_update does not support future exogenous inputs")
    model = model_builder()
    spec = getattr(model, "_model_spec", None)
    update = getattr(model, "update", None)
    if not callable(update) or (spec is not None and not spec.supports_update):
        raise ValueError("model does not support fixed-parameter update; use refit-per-step recursive instead")


def _recursive_with_update(
    model_builder: Callable[[], BaseStatModel],
    history_series: pd.Series,
    history_frame: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """recursive 的 forward 快速路径：首步 fit，后续步固定参数 update 滤波。

    与旧逐步重拟合语义不同（滤波 vs 重估计）；数值接近但不逐值相等，
    属预期差异。update 在增长历史上重新滤波，等价于回测 refit_every 的
    窗口间更新机制。
    """
    hist = history_series.copy()
    hist_frame = history_frame.copy()
    preds: list[float] = []
    model = None
    for step_idx in range(horizon):
        if step_idx == 0:
            model = model_builder()
            model.fit(hist, X_hist=hist_frame if hist_frame.shape[1] > 1 else None)
        else:
            assert model is not None
            update = getattr(model, "update", None)
            if not callable(update):
                raise ValueError("model does not support fixed-parameter update")
            update(hist, X_hist=hist_frame if hist_frame.shape[1] > 1 else None)
        assert model is not None
        next_val = _predict_one(model, None)
        preds.append(next_val)
        hist = pd.concat([hist, pd.Series([next_val])], ignore_index=True)
        hist.name = history_series.name
        hist_frame = _append_history_frame(hist_frame, next_val, None)
    return pd.Series(preds, name="yhat")


def checked_model_builder(model_builder, strategy, X_future, intervals=False, history=None, X_hist=None):
    """注册模型能力是执行门禁；自定义模型仍遵守公共接口。"""
    def build():
        """构建模型并按 registry 能力位校验「策略 × 输入 × 区间」组合。"""
        model = model_builder()
        spec = getattr(model, "_model_spec", None)
        if spec is not None:
            ignore = getattr(model, "_ignore_unsupported_inputs", False)
            if strategy == "native" and not spec.supports_native_multistep:
                raise ValueError("model does not support native multistep prediction")
            if X_future is not None and X_future.shape[1] and not spec.supports_future_exog and not ignore:
                raise ValueError("model does not support future exogenous inputs")
            if history is not None and not ignore and not (spec.supports_multivariate or spec.supports_future_exog):
                frame = combine_history_frame(history, X_hist)
                if frame.shape[1] > 1:
                    raise ValueError("model does not support historical covariates; set ignore_unsupported_inputs explicitly")
            if intervals and strategy == "native" and not spec.supports_prediction_intervals:
                raise ValueError("model does not support native intervals; use conformal intervals")
        return model
    return build


def _predict_direct_step(model, step: int, X_future_prefix: pd.DataFrame | None = None) -> float:
    pred = model.predict(step, X_future=X_future_prefix)
    series = pred if isinstance(pred, pd.Series) else pd.Series(pred)
    if len(series) != step:
        raise ValueError(f"predict({step}) returned length {len(series)}")
    return float(series.iloc[-1])


def run_point_inference(
    model_builder: Callable[[], BaseStatModel],
    history: pd.Series | pd.DataFrame,
    horizon: int,
    forecast_strategy: str,
    X_hist: pd.DataFrame | None = None,
    X_future: pd.DataFrame | None = None,
    use_update: bool = False,
    feature_context: FutureFeatures | None = None,
) -> pd.Series:
    """执行点预测的统一多步推理编排。

    native: 一次拟合预测完整 horizon；single_step: 只允许 horizon=1；
    direct: 每个预测步重新拟合一个模型并取对应步长的最后一个预测值；
    recursive: 每一步把上一轮预测追加回历史，再预测下一步；
    dirrec: 逐步重建模型，同时使用递归扩展后的历史。
    use_update=True 时 recursive 走 forward 快速路径：首步 fit，
    后续步固定参数 update 滤波（仅 supports_update 模型，无未来外生）。
    """
    strategy = normalize_forecast_strategy(forecast_strategy)
    validate_horizon(horizon)
    validate_single_step_horizon(strategy, horizon)
    if use_update:
        _validate_update_path(strategy, model_builder, X_future)

    if feature_context is not None:
        if use_update:
            raise ValueError("use_update does not support derived future covariates")
        return _run_feature_inference(model_builder, history, horizon, strategy, X_hist, X_future, feature_context)

    model_builder = checked_model_builder(model_builder, strategy, X_future, history=history, X_hist=X_hist)
    history_series = to_univariate_series(history).astype(float).reset_index(drop=True)
    history_frame = combine_history_frame(history, X_hist).astype(float).reset_index(drop=True)
    future_frame = None if X_future is None else to_dataframe(X_future).astype(float).reset_index(drop=True)

    if strategy == "native":
        model = model_builder()
        model.fit(history_series, X_hist=history_frame, X_future=future_frame)
        pred = pd.Series(model.predict(horizon, X_future=future_frame)).reset_index(drop=True)
        if len(pred) != horizon:
            raise ValueError(f"predict({horizon}) returned length {len(pred)}")
        return pred.rename("yhat")

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

    if strategy == "recursive" and use_update:
        return _recursive_with_update(model_builder, history_series, history_frame, horizon)

    hist = history_series.copy()
    hist_frame = history_frame.copy()
    preds = []
    for step_idx in range(horizon):
        # 保留旧语义：两者均每步重建并拟合；recursive 对新实例再复制，并不复用拟合参数。
        model = model_builder() if strategy == "dirrec" else copy.deepcopy(model_builder())
        next_future = _future_row(future_frame, step_idx)
        model.fit(hist, X_hist=hist_frame, X_future=next_future)
        next_val = _predict_one(model, next_future)
        preds.append(next_val)
        hist = pd.concat([hist, pd.Series([next_val])], ignore_index=True)
        hist.name = history_series.name
        hist_frame = _append_history_frame(hist_frame, next_val, next_future)
    return pd.Series(preds, name="yhat")


def _run_feature_inference(model_builder, history, horizon, strategy, X_hist, X_future,
                           context: FutureFeatures) -> pd.Series:
    """按预测推进派生协变量；native 只 fit 一次，direct 保持逐步重拟合。

    目标 lag 未知部分使用已生成的点预测；原生后端逐增前缀 predict，不冻结
    未知目标。原生 predict 调用次数增加，但不重复 fit。
    """
    frame = combine_history_frame(history, X_hist)
    first = context.row(X_future, 0, [])
    build = checked_model_builder(model_builder, strategy, first, history=history, X_hist=X_hist)
    if strategy in {"recursive", "dirrec", "single_step"}:
        preds: list[float] = []
        for step in range(horizon):
            future = context.row(X_future, step, preds)
            model = build()
            model.fit(frame.iloc[:, 0], X_hist=frame, X_future=future)
            preds.append(_predict_one(model, future))
            frame = _append_history_frame(frame, preds[-1], future)
        return pd.Series(preds, name="yhat")

    def predict_prefix(model, length):
        preds, rows = [], []
        for step in range(length):
            rows.append(context.row(X_future, step, preds))
            preds.append(_predict_direct_step(model, step + 1, pd.concat(rows, ignore_index=True)))
        return preds

    if strategy == "native":
        model = build()
        model.fit(frame.iloc[:, 0], X_hist=frame, X_future=first)
        return pd.Series(predict_prefix(model, horizon), name="yhat")
    preds = []
    for length in range(1, horizon + 1):
        model = build()
        model.fit(frame.iloc[:, 0], X_hist=frame, X_future=first)
        preds.append(predict_prefix(model, length)[-1])
    return pd.Series(preds, name="yhat")


def run_interval_inference(
    model_builder: Callable[[], BaseStatModel],
    history: pd.Series | pd.DataFrame,
    horizon: int,
    forecast_strategy: str,
    X_hist: pd.DataFrame | None = None,
    X_future: pd.DataFrame | None = None,
    alpha: float = 0.05,
    levels: Sequence[float] | None = None,
) -> pd.DataFrame:
    """执行区间预测编排。

    native/single_step/direct 从同一拟合模型同时取点预测与原生区间；
    多水平经 predict_with_levels 一次预测返回全部水平列（拟合次数不随水平数增加）；
    recursive/dirrec 下原生区间不可用（逐步重拟合破坏解析连续性），
    在此显式失败并指向 conformal，不再产出 NaN 区间列（P6 行为变更）。
    """
    strategy = normalize_forecast_strategy(forecast_strategy)
    validate_horizon(horizon)
    validate_single_step_horizon(strategy, horizon)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    # 延迟导入避免 intervals ↔ strategies 循环依赖（IntervalSpec 协议归 intervals）。
    from forecasting.intervals import iter_bound_pairs, resolve_interval_levels

    resolved_levels = resolve_interval_levels(levels, alpha)
    multi = len(resolved_levels) > 1
    if strategy in {"recursive", "dirrec"}:
        raise ValueError(
            f"native intervals are unavailable under strategy={strategy!r}; "
            "use interval_method=conformal instead"
        )
    model_builder = checked_model_builder(model_builder, strategy, X_future, intervals=True, history=history, X_hist=X_hist)

    history_series = to_univariate_series(history).astype(float).reset_index(drop=True)
    history_frame = combine_history_frame(history, X_hist).astype(float).reset_index(drop=True)
    future_frame = None if X_future is None else to_dataframe(X_future).astype(float).reset_index(drop=True)

    if strategy in {"single_step", "native"}:
        model = model_builder()
        first_future = future_frame if strategy == "native" else _future_row(future_frame, 0)
        model.fit(history_series, X_hist=history_frame, X_future=first_future)
        result = model.predict_with_levels(horizon, X_future=first_future, levels=resolved_levels)
        if len(result) != horizon:
            raise ValueError(f"interval prediction length {len(result)} != horizon {horizon}")
        if strategy == "native":
            bound_cols = [c for c in result.columns if c.startswith(("yhat_lower", "yhat_upper"))]
            values = result[["yhat", *bound_cols]].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("invalid native intervals; use conformal or inspect backend failure")
            for lower_col, upper_col in iter_bound_pairs(result):
                if (result[lower_col] > result[upper_col]).any():
                    raise ValueError("invalid native intervals; use conformal or inspect backend failure")
        return result.reset_index(drop=True)

    rows: list[dict[str, float]] = []
    for step_idx in range(horizon):
        model = model_builder()
        prefix = _future_prefix(future_frame, step_idx + 1)
        model.fit(history_series, X_hist=history_frame, X_future=prefix)
        result = model.predict_with_levels(step_idx + 1, X_future=prefix, levels=resolved_levels).reset_index(drop=True)
        if len(result) != step_idx + 1:
            raise ValueError("interval prediction length mismatch")
        row: dict[str, float] = {"yhat": float(result["yhat"].iloc[-1])}
        last = result.iloc[-1]
        for lower_col, upper_col in iter_bound_pairs(result):
            row[lower_col] = float(last[lower_col]) if pd.notna(last[lower_col]) else np.nan
            row[upper_col] = float(last[upper_col]) if pd.notna(last[upper_col]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
