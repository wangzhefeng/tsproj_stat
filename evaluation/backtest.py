from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import pandas as pd

from .metrics import bias, mae, mape, max_error, mse, r2, rmse, smape
from utils.log_util import logger


@dataclass
class BacktestResult:
    predictions_df: pd.DataFrame
    metrics_df: pd.DataFrame
    summary_df: pd.DataFrame
    summary: dict[str, float | int]
    failed_windows: list[dict]


def _run_single_window(args: tuple) -> dict | None:
    """Execute one backtest window. Returns result dict or None on failure.

    Top-level function (not a method) so it is picklable for multiprocessing.
    """
    (
        window_id,
        train_end,
        train_y,
        test_y,
        train_x_hist,
        test_x_future,
        test_slice_time,  # pd.Series or None (timestamps for time_col)
        model,
        horizon,
        time_col,
    ) = args
    try:
        model_copy = copy.deepcopy(model)
        model_copy.fit(train_y, X_hist=train_x_hist, X_future=test_x_future)
        pred = model_copy.predict(horizon, X_future=test_x_future).astype(float).reset_index(drop=True)
    except Exception as exc:
        return {
            "status": "failed",
            "window_id": int(window_id),
            "train_end": int(train_end),
            "error": str(exc),
        }

    residual = test_y - pred
    metric_row = {
        "window_id": int(window_id),
        "train_end": int(train_end),
        "horizon": int(horizon),
        "mae": mae(test_y.values, pred.values),
        "rmse": rmse(test_y.values, pred.values),
        "mape": mape(test_y.values, pred.values),
        "smape": smape(test_y.values, pred.values),
        "mse": mse(test_y.values, pred.values),
        "r2": r2(test_y.values, pred.values),
        "bias": bias(test_y.values, pred.values),
        "max_error": max_error(test_y.values, pred.values),
    }
    prediction_rows = []
    for idx in range(horizon):
        row: dict = {
            "window_id": int(window_id),
            "train_end": int(train_end),
            "horizon_step": int(idx + 1),
            "y_true": float(test_y.iloc[idx]),
            "y_pred": float(pred.iloc[idx]),
            "residual": float(residual.iloc[idx]),
        }
        if time_col is not None and test_slice_time is not None:
            row["timestamp"] = test_slice_time.iloc[idx]
        prediction_rows.append(row)

    return {
        "status": "ok",
        "window_id": int(window_id),
        "train_end": int(train_end),
        "metric_row": metric_row,
        "prediction_rows": prediction_rows,
    }


def rolling_backtest(
    df: pd.DataFrame,
    model,
    target_col: str = "y",
    time_col: str | None = None,
    endog_cols: list[str] | None = None,
    exog_cols: list[str] | None = None,
    future_exog_cols: list[str] | None = None,
    initial_train_size: int = 30,
    horizon: int = 7,
    step: int = 7,
    verbose: bool = False,
    progress_every: int = 10,
    n_jobs: int = 1,
) -> BacktestResult:
    n = len(df)
    if initial_train_size + horizon > n:
        raise ValueError("Not enough data for backtest")
    if progress_every <= 0:
        raise ValueError("progress_every must be > 0")
    endog_cols = endog_cols or [target_col]
    exog_cols = exog_cols or []
    future_exog_cols = future_exog_cols or []
    feature_cols = []
    for col in [*endog_cols, *exog_cols]:
        if col != target_col and col not in feature_cols:
            feature_cols.append(col)
    future_cols = [col for col in future_exog_cols if col in df.columns]

    # Build argument list for all windows
    args_list: list[tuple] = []
    start = initial_train_size
    window_id = 0
    total_windows = ((n - initial_train_size - horizon) // step) + 1

    while start + horizon <= n:
        window_id += 1
        train_y = df[target_col].iloc[:start].copy()
        train_feature_cols = [target_col, *feature_cols]
        train_x_hist = None
        if any(col in df.columns for col in train_feature_cols):
            available_cols = [col for col in train_feature_cols if col in df.columns]
            if len(available_cols) > 1:
                train_x_hist = df[available_cols].iloc[:start].reset_index(drop=True)
        test_slice = df.iloc[start: start + horizon].reset_index(drop=True)
        test_y = test_slice[target_col].astype(float).reset_index(drop=True)
        test_x_future = None
        if future_cols:
            test_x_future = test_slice[future_cols].reset_index(drop=True)
        test_slice_time = test_slice[time_col] if (time_col is not None and time_col in test_slice.columns) else None

        args_list.append((
            window_id,
            start,
            train_y,
            test_y,
            train_x_hist,
            test_x_future,
            test_slice_time,
            model,
            horizon,
            time_col,
        ))
        start += step

    # Execute windows — parallel or sequential
    started_at = time.perf_counter()
    if n_jobs == 1 or n_jobs == 0:
        raw_results = []
        for i, args in enumerate(args_list):
            raw_results.append(_run_single_window(args))
            if verbose and (i + 1) % progress_every == 0:
                elapsed = time.perf_counter() - started_at
                w_idx = args[0]
                train_end = args[1]
                msg = (
                    f"[backtest] window {w_idx}/{total_windows} "
                    f"train_end={train_end} total_seconds={elapsed:.3f}"
                )
                print(msg)
                logger.info(msg)
    else:
        from concurrent.futures import ProcessPoolExecutor
        max_workers = n_jobs if n_jobs > 0 else None
        logger.info(f"[Backtest] running {total_windows} windows with n_jobs={n_jobs}")
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                raw_results = list(pool.map(_run_single_window, args_list))
        except Exception as exc:
            logger.warning(f"[Backtest] parallel execution failed ({exc}), falling back to sequential")
            raw_results = [_run_single_window(a) for a in args_list]

    # Aggregate results
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    failed_windows: list[dict] = []

    for res in raw_results:
        if res is None:
            continue
        if res["status"] == "failed":
            failed_windows.append({
                "window_id": res["window_id"],
                "train_end": res["train_end"],
                "error": res["error"],
            })
            logger.warning(
                f"[Backtest] window {res['window_id']}/{total_windows} "
                f"failed (train_end={res['train_end']}): {res['error']}"
            )
        else:
            metric_rows.append(res["metric_row"])
            prediction_rows.extend(res["prediction_rows"])

    if failed_windows:
        logger.warning(
            f"[Backtest] {len(failed_windows)}/{total_windows} windows failed and were skipped"
        )

    if not metric_rows:
        raise RuntimeError("All backtest windows failed — no metrics to report")

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    summary_values = {
        "window_count": int(len(metrics_df)),
        "failed_windows": int(len(failed_windows)),
        "horizon": int(horizon),
        "mae": float(metrics_df["mae"].mean()),
        "rmse": float(metrics_df["rmse"].mean()),
        "mape": float(metrics_df["mape"].mean()),
        "smape": float(metrics_df["smape"].mean()),
        "mse": float(metrics_df["mse"].mean()),
        "r2": float(metrics_df["r2"].mean()),
        "bias": float(metrics_df["bias"].mean()),
        "max_error": float(metrics_df["max_error"].mean()),
    }
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
