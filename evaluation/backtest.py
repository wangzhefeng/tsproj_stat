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
import numpy as np

from forecasting.strategies import normalize_forecast_strategy, normalize_window_mode, run_point_inference
from .metrics import bias, mae, mape, max_error, mse, r2, rmse, smape
from .metrics import coverage, interval_width, winkler_score
from forecasting.intervals import iter_bound_pairs, predict_frame
from forecasting.strategies import checked_model_builder
from models.base import BaseStatModel
from data_provider.target_transforms.transformer import TargetTransformer
from data_provider.cleaning.imputation import repair_history_frame, require_finite
from utils.log_util import logger


@dataclass
class BacktestResult:
    """rolling backtest 的结构化返回。

    predictions_df 保存逐窗口逐步预测，metrics_df 保存窗口级指标，
    summary_df/summary 保存跨窗口汇总，failed_windows 记录被跳过的失败窗口。
    """
    predictions_df: pd.DataFrame
    metrics_df: pd.DataFrame
    summary_df: pd.DataFrame
    summary: dict[str, float | int | str]
    failed_windows: list[dict]


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
) -> BacktestResult:
    """执行滚动回测。

    expanding 窗口从序列开头逐步扩张；sliding 窗口保持固定 train_size。
    levels 支持多置信水平区间指标（列名带水平后缀，如 interval_coverage_80）。
    默认 RAISE：任一窗口失败即中止（避免汇总指标带存活偏差）；
    显式 allow_failed_windows=True 时才跳过失败窗口，且 summary 打标 survivor_bias。
    """
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
            if interval_method == "none":
                # 原点之后的观测不可见；区间路径由每个内部校准原点独立修复。
                repaired, audit = repair_history_frame(train_slice, available_hist_cols)
                repaired_count = audit.filled_value_count
                train_y = repaired[target_col].astype(float).reset_index(drop=True)
                if train_x_hist is not None:
                    train_x_hist = repaired[available_hist_cols].astype(float).reset_index(drop=True)
            if proc is not None and use_proc and interval_method == "none":
                train_y_model = proc.fit_transform(train_y)
                if train_x_hist is not None and target_col in train_x_hist.columns:
                    train_x_hist = train_x_hist.copy()
                    train_x_hist[target_col] = train_y_model.values
            else:
                train_y_model = train_y
            # 每个回测窗口都通过统一推理入口运行，保证 test 与 forecast 策略一致。
            if refit_every != 1:
                did_refit = cached_model is None or (refit_every > 0 and (window_id - 1) % refit_every == 0)
                if cached_model is None or (refit_every > 0 and (window_id - 1) % refit_every == 0):
                    cached_model = checked_model_builder(model_builder, strategy, test_x_future)()
                    cached_model.fit(train_y, X_hist=train_x_hist, X_future=test_x_future)
                else:
                    update = getattr(cached_model, "update", None)
                    if not callable(update):
                        raise ValueError("model does not support fixed-parameter update")
                    update(train_y, X_hist=train_x_hist)
                pred = cached_model.predict(horizon, X_future=test_x_future).reset_index(drop=True)
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
                ).astype(float).reset_index(drop=True)
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
        metric_row = {
            "window_id": int(window_id),
            "train_start": int(train_start),
            "train_end": int(start),
            "horizon": int(horizon),
            "mae": mae(test_y.values, pred.values),
            "rmse": rmse(test_y.values, pred.values),
            "mape": mape(test_y.values, pred.values),
            "smape": smape(test_y.values, pred.values),
            "mse": mse(test_y.values, pred.values),
            "r2": r2(test_y.values, pred.values),
            "bias": bias(test_y.values, pred.values),
            "max_error": max_error(test_y.values, pred.values),
            "refitted": did_refit,
            "history_filled_value_count": repaired_count if interval_method == "none" else None,
            "history_repair_policy": "window_local_linear" if interval_method == "none" else "per_calibration_origin",
        }
        pred_rows = []
        if interval_df is not None:
            bound_pairs = [(c, u) for c, u in iter_bound_pairs(interval_df)]
            for lower_col, upper_col in bound_pairs:
                suffix = lower_col[len("yhat_lower"):]
                level_tag = suffix.strip("_") if suffix else None
                coverage_key = f"interval_coverage_{level_tag}" if level_tag else "interval_coverage"
                width_key = f"interval_width_{level_tag}" if level_tag else "interval_width"
                winkler_key = f"winkler_score_{level_tag}" if level_tag else "winkler_score"
                lower, upper = interval_df[lower_col], interval_df[upper_col]
                level_alpha = 1.0 - (float(level_tag) / 100.0) if level_tag else interval_alpha
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
        return {"failed": False, "metric_row": metric_row, "prediction_rows": pred_rows}

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
        "mae": float(metrics_df["mae"].mean()),
        "rmse": float(metrics_df["rmse"].mean()),
        "mape": float(metrics_df["mape"].mean()),
        "smape": float(metrics_df["smape"].mean()),
        "mse": float(metrics_df["mse"].mean()),
        "r2": float(metrics_df["r2"].mean()),
        "bias": float(metrics_df["bias"].mean()),
        "max_error": float(metrics_df["max_error"].mean()),
    }
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
        summary_df=summary_df,
        summary=summary_values,
        failed_windows=failed_windows,
    )
