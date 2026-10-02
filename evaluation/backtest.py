"""rolling backtest：窗口切分、逐窗推理、窗口级指标与跨窗口汇总。

默认 RAISE 语义：任一窗口失败即中止；显式 allow_failed_windows 才跳过
并在 summary 打标 survivor_bias。区间指标按置信水平展开（interval_coverage_80 式）。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from features.model_inputs import ModelFeatureSpec
import numpy as np

from forecasting.strategies import normalize_forecast_strategy, normalize_window_mode, run_point_inference
from forecasting.origins import prepare_origin_inputs
from .metrics import POINT_METRICS, train_scales
from .metrics import coverage, interval_width, winkler_score
from forecasting.intervals import iter_bound_pairs, predict_frame
from models.contracts.intervals import resolve_interval_levels, interval_bound_columns
from forecasting.strategies import checked_model_builder
from models.base import BaseStatModel
from data_provider.target_transforms.transformer import TargetTransformer
from data_provider.quality.checks import require_finite, require_regular_time
from utils.log_util import logger


@dataclass
class BacktestResult:
    """rolling backtest 的结构化返回。

    predictions_df 保存逐窗口逐步预测，metrics_df 保存窗口级指标，
    step_metrics_df 保存按 horizon_step 跨窗聚合的指标（含逐窗缩放后的 mase/rmsse），
    summary_df/summary 保存跨窗口汇总，failed_windows 记录被跳过的失败窗口。
    """
    predictions_df: pd.DataFrame
    metrics_df: pd.DataFrame
    step_metrics_df: pd.DataFrame
    summary_df: pd.DataFrame
    summary: dict[str, float | int | str]
    failed_windows: list[dict]


def _build_step_metrics(
    predictions_df: pd.DataFrame,
    window_scales: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """按 horizon_step 跨窗口聚合点指标。

    mase/rmsse 按逐窗训练缩放基准逐点缩放后聚合（pool 语义）：
    mase = mean(|e|/scale_w)，rmsse = sqrt(mean(e²/scale_sq_w))；
    缩放基准不可用的窗口跳过，全缺为 NaN。
    """
    scales_df = pd.DataFrame(
        [(wid, scales[0], scales[1]) for wid, scales in window_scales.items()],
        columns=["window_id", "_mase_scale", "_rmsse_scale"],
    )
    merged = predictions_df.merge(scales_df, on="window_id", how="left")
    rows: list[dict] = []
    for step in sorted(merged["horizon_step"].unique()):
        grp = merged[merged["horizon_step"] == step]
        y_true = grp["y_true"].to_numpy(dtype=float)
        y_pred = grp["y_pred"].to_numpy(dtype=float)
        row: dict[str, float | int] = {"horizon_step": int(step), "window_count": int(len(grp))}
        for metric_name, metric_spec in POINT_METRICS.items():
            if metric_spec.requires_train:
                continue
            row[metric_name] = metric_spec.func(y_true, y_pred)
        err = y_true - y_pred
        mase_scale = grp["_mase_scale"].to_numpy(dtype=float)
        valid = np.isfinite(mase_scale) & (mase_scale > 0)
        row["mase"] = float(np.mean(np.abs(err[valid]) / mase_scale[valid])) if valid.any() else float("nan")
        rmsse_scale = grp["_rmsse_scale"].to_numpy(dtype=float)
        valid_sq = np.isfinite(rmsse_scale) & (rmsse_scale > 0)
        row["rmsse"] = (
            float(np.sqrt(np.mean((err[valid_sq] ** 2) / rmsse_scale[valid_sq])))
            if valid_sq.any() else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_backtest(
    df: pd.DataFrame,
    model_builder: Callable[[], BaseStatModel],
    target_col: str = "y",
    time_col: str | None = None,
    endog_cols: list[str] | None = None,
    exog_cols: list[str] | None = None,
    future_exog_cols: list[str] | None = None,
    train_size: int = 30,
    horizon: int = 7,
    step: int = 7,
    forecast_strategy: str = "direct",
    window_mode: str = "expanding",
    verbose: bool = False,
    progress_every: int = 10,
    n_jobs: int = 1,
    processor_builder: Callable[[], TargetTransformer] | None = None,
    allow_failed_windows: bool = False,
    interval_method: str = "none",
    interval_alpha: float = 0.05,
    conformal_n_windows: int = 20,
    levels: list[float] | None = None,
    refit_every: int = 1,
    feature_spec: ModelFeatureSpec | None = None,
) -> BacktestResult:
    """执行滚动回测。

    expanding 窗口从序列开头逐步扩张；sliding 窗口保持固定 train_size。
    levels 支持多置信水平区间指标（列名带水平后缀，如 interval_coverage_80）。
    默认 RAISE：任一窗口失败即中止（避免汇总指标带存活偏差）；
    显式 allow_failed_windows=True 时才跳过失败窗口，且 summary 打标 survivor_bias。
    """
    if time_col is not None:
        require_regular_time(df[time_col], role="backtest")
    n = len(df)
    if min(train_size, horizon, step) <= 0:
        raise ValueError("train_size, horizon and step must be positive")
    if isinstance(refit_every, bool) or not isinstance(refit_every, int) or refit_every < 0:
        raise ValueError("refit_every must be an integer >= 0")
    if train_size + horizon > n:
        raise ValueError("Not enough data for backtest")
    if progress_every <= 0:
        raise ValueError("progress_every must be > 0")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be > 0")

    strategy = normalize_forecast_strategy(forecast_strategy)
    resolved_window_mode = normalize_window_mode(window_mode)
    endog_cols = endog_cols or []
    exog_cols = exog_cols or []
    future_exog_cols = future_exog_cols or []

    feature_cols = []
    for col in [*endog_cols, *exog_cols]:
        if col != target_col and col not in feature_cols:
            feature_cols.append(col)
    future_cols = [col for col in future_exog_cols if col in df.columns]
    # 回测的未来外生直接取 df 真实值 = perfect foresight；必须在 summary 中披露（T16）。
    future_exog_policy = "perfect_foresight" if future_cols else "none"
    cached_model = None
    if refit_every != 1:
        if feature_spec is not None:
            raise ValueError("fixed-parameter update does not support derived future covariates")
        if strategy != "native" or n_jobs != 1 or interval_method != "none":
            raise ValueError("fixed-parameter update requires native, n_jobs=1, no intervals")
        if processor_builder is not None and processor_builder().enabled:
            raise ValueError("fixed-parameter update cannot change preprocessing scale")
        candidate = model_builder()
        spec = getattr(candidate, "_model_spec", None)
        if not callable(getattr(candidate, "update", None)) or (spec is not None and not spec.supports_update):
            raise ValueError("model does not support fixed-parameter update")

    total_windows = ((n - train_size - horizon) // step) + 1
    started_at = time.perf_counter()
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    failed_windows: list[dict] = []
    window_scales: dict[int, tuple[float, float]] = {}

    windows = []
    start = train_size
    window_id = 0
    while start + horizon <= n:
        window_id += 1
        # expanding 使用全部历史，sliding 只保留最近 train_size 行。
        train_start = 0 if resolved_window_mode == "expanding" else start - train_size
        windows.append((window_id, train_start, start))
        start += step

    def evaluate_window(window: tuple[int, int, int]) -> dict:
        """执行单个回测窗口：切分 →（可选预处理）→ 推理 → 指标与逐步预测明细。"""
        nonlocal cached_model
        window_id, train_start, start = window
        train_slice = df.iloc[train_start:start].reset_index(drop=True)
        test_slice = df.iloc[start : start + horizon].reset_index(drop=True)

        train_y = train_slice[target_col].astype(float).reset_index(drop=True)
        train_x_hist = None
        train_feature_cols = [target_col, *feature_cols]
        available_hist_cols = [col for col in train_feature_cols if col in train_slice.columns]
        if len(available_hist_cols) > 1:
            train_x_hist = train_slice[available_hist_cols].astype(float).reset_index(drop=True)

        test_y = test_slice[target_col].astype(float).reset_index(drop=True)
        test_x_future = None
        if future_cols:
            test_x_future = test_slice[future_cols].astype(float).reset_index(drop=True)
        test_time = test_slice[time_col] if time_col is not None and time_col in test_slice.columns else None

        proc = processor_builder() if processor_builder is not None else None
        use_proc = proc is not None and getattr(proc, "enabled", False)
        did_refit = True
        interval_df = None
        bound_pairs: list[tuple[str, str]] = []
        repaired_count = 0
        try:
            require_finite(test_y, "evaluation target")
            if test_x_future is not None:
                require_finite(test_x_future, "future exogenous data")
            # interval_method=none：修复→预处理经 forecasting.origins 共享原语
            # （与 conformal/simulate 的校准原点同一实现）；区间路径不在此处修复，
            # 由 predict_frame 的每个内部校准原点独立修复。
            prepared = None
            if interval_method == "none":
                builder: Callable[[], TargetTransformer] | None = None
                if use_proc:
                    assert proc is not None
                    builder = lambda: proc
                prepared = prepare_origin_inputs(
                    train_y, train_x_hist, builder, feature_spec,
                    train_slice[time_col] if time_col else None, test_time)
                repaired_count = prepared.audit.filled_value_count
                train_y = prepared.y_raw.reset_index(drop=True)
                train_y_model = prepared.y_model.reset_index(drop=True)
                train_x_hist = prepared.X_hist_model
            else:
                train_y_model = train_y
            # 每个回测窗口都通过统一推理入口运行，保证 test 与 forecast 策略一致。
            if refit_every != 1:
                did_refit = cached_model is None or (refit_every > 0 and (window_id - 1) % refit_every == 0)
                if did_refit:
                    cached_model = checked_model_builder(model_builder, strategy, test_x_future)()
                    cached_model.fit(train_y, X_hist=train_x_hist, X_future=test_x_future)
                else:
                    update = getattr(cached_model, "update", None)
                    if not callable(update):
                        raise ValueError("model does not support fixed-parameter update")
                    update(train_y, X_hist=train_x_hist)
                # did_refit=False 蕴含 cached_model 非 None（首窗必 fit）。
                assert cached_model is not None
                pred = cached_model.predict(horizon, X_future=test_x_future)
            elif interval_method != "none":
                interval_df = predict_frame(model_builder, train_y, horizon, strategy,
                                            train_x_hist, test_x_future, processor_builder,
                                            interval_method, interval_alpha, conformal_n_windows,
                                            levels=levels)
                pred = interval_df["yhat"]
            else:
                pred = run_point_inference(
                    model_builder=model_builder,
                    history=train_y_model,
                    horizon=horizon,
                    forecast_strategy=strategy,
                    X_hist=train_x_hist,
                    X_future=test_x_future,
                    feature_context=prepared.feature_context if prepared is not None else None,
                )
            # 三条推理分支的统一归一化出口：dtype=float、0..h-1 索引。
            pred = pd.Series(pred, dtype=float).reset_index(drop=True)
            if proc is not None and use_proc and interval_method == "none":
                pred = proc.inverse_forecast(pred).astype(float).reset_index(drop=True)
            if len(pred) != horizon or not np.isfinite(pred.to_numpy()).all():
                raise ValueError("non-finite or incorrect-length backtest forecast")
        except Exception as exc:
            cached_model = None
            # 部分模型在个别窗口可能拟合失败；记录失败窗口，避免一个窗口拖垮整次评估。
            return {
                "failed": True,
                "failed_window": {
                    "window_id": int(window_id),
                    "train_start": int(train_start),
                    "train_end": int(start),
                    "error": str(exc),
                },
            }

        residual = test_y - pred
        # 点指标由注册表驱动；mase/rmsse 的缩放基准取当前窗口原始尺度训练序列。
        metric_row: dict[str, object] = {
            "window_id": int(window_id),
            "train_start": int(train_start),
            "train_end": int(start),
            "horizon": int(horizon),
        }
        for metric_name, metric_spec in POINT_METRICS.items():
            if metric_spec.requires_train:
                metric_row[metric_name] = metric_spec.func(test_y.values, pred.values, train_y.values)
            else:
                metric_row[metric_name] = metric_spec.func(test_y.values, pred.values)
        metric_row["refitted"] = did_refit
        metric_row["history_filled_value_count"] = repaired_count if interval_method == "none" else None
        metric_row["history_repair_policy"] = (
            "window_local_linear" if interval_method == "none" else "per_calibration_origin"
        )
        # 逐窗缩放基准随结果带出，供按 horizon_step 聚合 mase/rmsse 时逐点缩放。
        naive_scale, sq_scale = train_scales(train_y.values)
        pred_rows = []
        if interval_df is not None:
            bound_pairs = [(c, u) for c, u in iter_bound_pairs(interval_df)]
            resolved_levels = resolve_interval_levels(levels, interval_alpha)
            alphas = {interval_bound_columns(level, multi=len(resolved_levels) > 1)[0]: 1 - level
                      for level in resolved_levels}
            for lower_col, upper_col in bound_pairs:
                suffix = lower_col[len("yhat_lower"):]
                level_tag = suffix.strip("_") if suffix else None
                coverage_key = f"interval_coverage_{level_tag}" if level_tag else "interval_coverage"
                width_key = f"interval_width_{level_tag}" if level_tag else "interval_width"
                winkler_key = f"winkler_score_{level_tag}" if level_tag else "winkler_score"
                lower, upper = interval_df[lower_col], interval_df[upper_col]
                level_alpha = alphas[lower_col]
                metric_row.update(
                    {coverage_key: coverage(test_y, lower, upper),
                     width_key: interval_width(lower, upper),
                     winkler_key: winkler_score(test_y, lower, upper, level_alpha)},
                )
        for idx in range(horizon):
            row: dict[str, object] = {
                "window_id": int(window_id),
                "train_start": int(train_start),
                "train_end": int(start),
                "horizon_step": int(idx + 1),
                "y_true": float(test_y.iloc[idx]),
                "y_pred": float(pred.iloc[idx]),
                "residual": float(residual.iloc[idx]),
            }
            if test_time is not None:
                row["timestamp"] = test_time.iloc[idx]
            if interval_df is not None:
                for lower_col, upper_col in bound_pairs:
                    row[lower_col] = float(interval_df[lower_col].iloc[idx])
                    row[upper_col] = float(interval_df[upper_col].iloc[idx])
            pred_rows.append(row)
        return {
            "failed": False,
            "metric_row": metric_row,
            "prediction_rows": pred_rows,
            "train_scales": (naive_scale, sq_scale),
        }

    if n_jobs == 1:
        results = [evaluate_window(window) for window in windows]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(evaluate_window, windows))

    for window, result in zip(windows, results):
        window_id, train_start, start = window
        if result["failed"]:
            failed_window = result["failed_window"]
            if not allow_failed_windows:
                raise RuntimeError(
                    f"[Backtest] window {window_id}/{total_windows} failed "
                    f"(train_start={train_start}, train_end={start}): {failed_window['error']}"
                    "；如需容忍失败窗口，请显式设置 backtest_allow_failed_windows=true"
                )
            failed_windows.append(failed_window)
            logger.warning(
                f"[Backtest] window {window_id}/{total_windows} failed "
                f"(train_start={train_start}, train_end={start}): {failed_window['error']}"
            )
        else:
            metric_rows.append(result["metric_row"])
            prediction_rows.extend(result["prediction_rows"])
            window_scales[window_id] = result["train_scales"]

        if verbose and window_id % progress_every == 0:
            # 这里显式 print 到 stdout，满足 CLI smoke/test 对终端进度可见性的要求。
            elapsed = time.perf_counter() - started_at
            msg = (
                f"[backtest] window {window_id}/{total_windows} "
                f"train_start={train_start} train_end={start} total_seconds={elapsed:.3f}"
            )
            print(msg)
            logger.info(msg)

    if failed_windows:
        logger.warning(f"[Backtest] {len(failed_windows)}/{total_windows} windows failed and were skipped")
    if not metric_rows:
        raise RuntimeError("All backtest windows failed — no metrics to report")

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    step_metrics_df = _build_step_metrics(predictions_df, window_scales)
    summary_values: dict[str, float | int | str] = {
        # 汇总指标采用窗口级指标均值，窗口级明细仍保留在 metrics_df 中。
        # survivor_bias 打标：存在被跳过的失败窗口时，汇总指标只在成功窗口上平均。
        "window_count": int(len(metrics_df)),
        "failed_windows": int(len(failed_windows)),
        "survivor_bias": bool(failed_windows),
        "horizon": int(horizon),
        "train_size": int(train_size),
        "step": int(step),
        "window_mode": resolved_window_mode,
        "forecast_strategy": strategy,
        "future_exog_policy": future_exog_policy,
    }
    # 点指标均值由注册表驱动；NaN 窗口（如常数训练窗的 mase/rmsse）跳过，全缺为 NaN。
    for metric_name in POINT_METRICS:
        values = metrics_df[metric_name].to_numpy(dtype=float)
        available = values[~np.isnan(values)]
        summary_values[metric_name] = float(available.mean()) if available.size else float("nan")
    summary_values["interval_method"] = interval_method
    summary_values["refit_every"] = refit_every
    summary_values["refit_count"] = int(metrics_df["refitted"].to_numpy(dtype=bool).sum())
    if interval_method != "none":
        interval_keys = [k for k in metrics_df.columns
                         if k.startswith(("interval_coverage", "interval_width", "winkler_score"))]
        for key in interval_keys:
            values = metrics_df[key].to_numpy(dtype=float)
            available = values[~np.isnan(values)]
            summary_values[key] = float(available.mean()) if available.size else float("nan")
    summary_df = pd.DataFrame([summary_values])
    total_elapsed = time.perf_counter() - started_at
    logger.info(f"[Backtest] done: {len(metrics_df)} windows in {total_elapsed:.1f}s")
    return BacktestResult(
        predictions_df=predictions_df,
        metrics_df=metrics_df,
        step_metrics_df=step_metrics_df,
        summary_df=summary_df,
        summary=summary_values,
        failed_windows=failed_windows,
    )
