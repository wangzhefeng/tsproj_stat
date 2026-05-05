from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from models.inference import normalize_inference_strategy, normalize_window_mode, run_point_inference
from .metrics import bias, mae, mape, max_error, mse, r2, rmse, smape
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
    model_builder: Callable[[], object],
    target_col: str = "y",
    time_col: str | None = None,
    endog_cols: list[str] | None = None,
    exog_cols: list[str] | None = None,
    future_exog_cols: list[str] | None = None,
    train_size: int = 30,
    horizon: int = 7,
    step: int = 7,
    inference_strategy: str = "direct",
    window_mode: str = "expanding",
    verbose: bool = False,
    progress_every: int = 10,
    n_jobs: int = 1,
) -> BacktestResult:
    """执行滚动回测。

    expanding 窗口从序列开头逐步扩张；sliding 窗口保持固定 train_size。
    单个窗口失败时记录错误并跳过，只要仍有成功窗口就继续产出评估结果。
    """
    n = len(df)
    if train_size + horizon > n:
        raise ValueError("Not enough data for backtest")
    if progress_every <= 0:
        raise ValueError("progress_every must be > 0")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be > 0")

    strategy = normalize_inference_strategy(inference_strategy, None)
    resolved_window_mode = normalize_window_mode(window_mode)
    endog_cols = endog_cols or [target_col]
    exog_cols = exog_cols or []
    future_exog_cols = future_exog_cols or []

    feature_cols = []
    for col in [*endog_cols, *exog_cols]:
        if col != target_col and col not in feature_cols:
            feature_cols.append(col)
    future_cols = [col for col in future_exog_cols if col in df.columns]

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

        try:
            # 每个回测窗口都通过统一推理入口运行，保证 test 与 forecast 策略一致。
            pred = run_point_inference(
                model_builder=model_builder,
                history=train_y,
                horizon=horizon,
                inference_strategy=strategy,
                X_hist=train_x_hist,
                X_future=test_x_future,
            ).astype(float).reset_index(drop=True)
        except Exception as exc:
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
        }
        pred_rows = []
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
        "window_count": int(len(metrics_df)),
        "failed_windows": int(len(failed_windows)),
        "horizon": int(horizon),
        "train_size": int(train_size),
        "step": int(step),
        "window_mode": resolved_window_mode,
        "inference_strategy": strategy,
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
