from __future__ import annotations

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


def rolling_backtest(
    df: pd.DataFrame,
    model,
    target_col: str = "y",
    time_col: str | None = None,
    initial_train_size: int = 30,
    horizon: int = 7,
    step: int = 7,
    verbose: bool = False,
    progress_every: int = 10,
) -> BacktestResult:
    n = len(df)
    if initial_train_size + horizon > n:
        raise ValueError("Not enough data for backtest")
    if progress_every <= 0:
        raise ValueError("progress_every must be > 0")

    metric_rows: list[dict[str, float | int]] = []
    prediction_rows: list[dict[str, object]] = []
    start = initial_train_size
    window_id = 0
    total_windows = ((n - initial_train_size - horizon) // step) + 1
    started_at = time.perf_counter()

    while start + horizon <= n:
        window_id += 1
        window_started_at = time.perf_counter()
        train_y = df[target_col].iloc[:start]
        test_slice = df.iloc[start : start + horizon].reset_index(drop=True)
        test_y = test_slice[target_col].astype(float)
        model.fit(train_y)
        pred = model.predict(horizon).astype(float).reset_index(drop=True)
        residual = test_y - pred

        metric_rows.append(
            {
                "window_id": int(window_id),
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
            }
        )

        for idx in range(horizon):
            row: dict[str, object] = {
                "window_id": int(window_id),
                "train_end": int(start),
                "horizon_step": int(idx + 1),
                "y_true": float(test_y.iloc[idx]),
                "y_pred": float(pred.iloc[idx]),
                "residual": float(residual.iloc[idx]),
            }
            if time_col is not None and time_col in test_slice.columns:
                row["timestamp"] = test_slice[time_col].iloc[idx]
            prediction_rows.append(row)
        if verbose and (
            window_id == 1
            or window_id == total_windows
            or window_id % progress_every == 0
        ):
            elapsed = time.perf_counter() - window_started_at
            total_elapsed = time.perf_counter() - started_at
            message = (
                f"[backtest] window {window_id}/{total_windows} "
                f"train_end={start} window_seconds={elapsed:.3f} total_seconds={total_elapsed:.3f}"
            )
            print(message)
            logger.info(message)
        start += step

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    summary_values = {
        "window_count": int(len(metrics_df)),
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
    return BacktestResult(
        predictions_df=predictions_df,
        metrics_df=metrics_df,
        summary_df=summary_df,
        summary=summary_values,
    )
