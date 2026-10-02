"""预测原语层：单一原点编排与滚动起点误差池。

forecast_at_origin 是 intervals（区间）与 simulation（路径模拟）共同的
推理编排：窗口内修复 → 预处理 → 多步推理 → 逆变换回原始尺度；
rolling_error_pool 是两者共同的校准循环（带符号误差，互不重叠验证段）。
本模块只做编排原语，不做区间列与模拟产物语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from data_provider.cleaning.imputation import RepairAudit, repair_history_frame
from data_provider.quality.checks import require_finite
from data_provider.target_transforms.transformer import TargetTransformer
from models.contracts.inputs import combine_history_frame
from forecasting.strategies import run_interval_inference, run_point_inference
from features.model_inputs import ModelFeatureSpec, FutureFeatures


@dataclass(frozen=True)
class PreparedOriginInputs:
    """prepare_origin_inputs 的产物：单一原点「修复→预处理」后的建模输入。

    y_raw 是修复后原始尺度序列（mase/rmsse 缩放基准等消费方使用）；
    y_model 是进一步经预处理变换的建模序列；X_hist_model 的目标列与
    y_model 同步变换、协变量列保持原尺度；processor 带出供逆变换复用。
    """

    y_raw: pd.Series
    y_model: pd.Series
    X_hist_model: pd.DataFrame | None
    processor: TargetTransformer | None
    audit: RepairAudit
    feature_context: FutureFeatures | None = None


def prepare_origin_inputs(
    y: pd.Series,
    X_hist: pd.DataFrame | None,
    processor_builder: Callable[[], TargetTransformer] | None,
    feature_spec: ModelFeatureSpec | None = None,
    history_time: pd.Series | None = None,
    future_time: pd.Series | None = None,
) -> PreparedOriginInputs:
    """修复窗口内缺失并对目标执行可逆预处理（不接触原点之后的数据）。

    forecast_at_origin 与回测 interval_method=none 路径的共享原语：
    先对合并帧逐列双向线性修复，再仅对目标列 fit_transform。
    """
    frame = combine_history_frame(y, X_hist)
    frame, audit = repair_history_frame(frame, list(frame.columns))
    y_raw = frame.iloc[:, 0]
    proc = processor_builder() if processor_builder is not None else None
    if proc is not None and proc.enabled:
        y_model = proc.fit_transform(y_raw)
        frame.iloc[:, 0] = y_model.to_numpy()
    else:
        y_model = y_raw
    feature_context = None
    if feature_spec is not None:
        frame, feature_context, _ = feature_spec.prepare(frame, history_time, future_time)
        y_model = frame.iloc[:, 0]
    return PreparedOriginInputs(
        y_raw=y_raw,
        y_model=y_model,
        X_hist_model=frame if X_hist is not None or feature_spec is not None else None,
        processor=proc,
        audit=audit,
        feature_context=feature_context,
    )


def forecast_at_origin(
    model_builder: Callable,
    y: pd.Series,
    *,
    h: int,
    strategy: str,
    X_hist: pd.DataFrame | None = None,
    X_future: pd.DataFrame | None = None,
    processor_builder: Callable | None = None,
    native_intervals: bool = False,
    alpha: float = 0.05,
    levels: Sequence[float] | None = None,
    feature_spec: ModelFeatureSpec | None = None,
    history_time: pd.Series | None = None,
    future_time: pd.Series | None = None,
) -> pd.DataFrame:
    """在单一原点执行一次预测：输入建模前 history，返回原始尺度结果。

    native_intervals=True 时返回点预测+区间列（一次拟合），否则仅 yhat；
    预处理启用时区间边界与点预测一起逆变换（乘法变换可能交换上下界）。
    """
    if native_intervals and feature_spec is not None:
        raise ValueError("native intervals do not support derived features")
    prepared = prepare_origin_inputs(y, X_hist, processor_builder, feature_spec, history_time, future_time)
    proc = prepared.processor
    if X_future is not None:
        require_finite(X_future, "future exogenous data")
    if native_intervals:
        result = run_interval_inference(model_builder, prepared.y_model, h, strategy, prepared.X_hist_model, X_future, alpha, levels)
    else:
        result = run_point_inference(model_builder, prepared.y_model, h, strategy, prepared.X_hist_model,
                                     X_future, feature_context=prepared.feature_context).to_frame("yhat")
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
    result.attrs["history_filled_value_count"] = prepared.audit.filled_value_count
    result.attrs["history_repair_policy"] = prepared.audit.policy
    return result


def rolling_error_pool(
    model_builder: Callable,
    y: pd.Series,
    x: pd.DataFrame,
    *,
    horizon: int,
    strategy: str,
    future_columns: list[str],
    processor_builder: Callable | None = None,
    first_origin: int,
    feature_spec: ModelFeatureSpec | None = None,
    history_time: pd.Series | None = None,
) -> np.ndarray:
    """滚动起点校准：返回带符号误差池 (n_windows, horizon)。

    与 conformal 区间、样本路径模拟共用的唯一校准实现：每个原点独立
    修复+预处理（不接触原点之后的数据），逐步带符号误差按行堆叠。
    有限值检查与历史充足性下限由调用方按各自语义报错。
    """
    scores: list[np.ndarray] = []
    for origin in range(first_origin, len(y), horizon):
        future = x.iloc[origin:origin + horizon][future_columns].reset_index(drop=True) if future_columns else None
        pred = forecast_at_origin(
            model_builder, y.iloc[:origin],
            h=horizon, strategy=strategy,
            X_hist=x.iloc[:origin], X_future=future,
            processor_builder=processor_builder, native_intervals=False,
            feature_spec=feature_spec,
            history_time=history_time.iloc[:origin] if history_time is not None else None,
            future_time=history_time.iloc[origin:origin + horizon].reset_index(drop=True) if history_time is not None else None,
        )
        scores.append(y.iloc[origin:origin + horizon].to_numpy() - pred.yhat.to_numpy())
    return np.asarray(scores)
