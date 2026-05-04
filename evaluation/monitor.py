from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import mae, rmse, mape

import os
LOGGING_LABEL = "monitor"
os.environ.setdefault("LOG_NAME", LOGGING_LABEL)
from utils.log_util import logger


class ModelMonitor:
    """Track rolling forecast quality and detect model degradation.

    Directory layout under monitor_dir/setting/:
        predictions_log.csv   — one row per forecast step logged
        actuals_log.csv       — actual values backfilled offline
        metrics_history.csv   — aggregated metrics snapshots
    """

    _PRED_COLS = ["run_id", "forecast_ts", "horizon_step", "yhat", "yhat_lower", "yhat_upper"]
    _ACT_COLS = ["forecast_ts", "horizon_step", "y_true"]
    _METRICS_COLS = ["snapshot_ts", "run_id", "window", "mae", "rmse", "mape"]

    def __init__(self, monitor_dir: str | Path, setting: str, window: int = 30):
        """
        Args:
            monitor_dir: root directory for monitor files
            setting: subdirectory name (typically model-data-pred_method)
            window: number of most recent matched actuals to use for rolling metrics
        """
        if window <= 0:
            raise ValueError("window must be > 0")
        self.monitor_dir = Path(monitor_dir) / setting
        self.window = window
        self._pred_path = self.monitor_dir / "predictions_log.csv"
        self._act_path = self.monitor_dir / "actuals_log.csv"
        self._metrics_path = self.monitor_dir / "metrics_history.csv"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_headers()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_forecast(
        self,
        run_id: str,
        yhat: pd.Series,
        yhat_lower: pd.Series | None = None,
        yhat_upper: pd.Series | None = None,
        forecast_ts: str | None = None,
    ) -> None:
        """Append a forecast to predictions_log.csv."""
        ts = forecast_ts or datetime.utcnow().isoformat(timespec="seconds") + "Z"
        rows = []
        for step, val in enumerate(yhat, start=1):
            rows.append({
                "run_id": run_id,
                "forecast_ts": ts,
                "horizon_step": step,
                "yhat": float(val),
                "yhat_lower": float(yhat_lower.iloc[step - 1]) if yhat_lower is not None else "",
                "yhat_upper": float(yhat_upper.iloc[step - 1]) if yhat_upper is not None else "",
            })
        self._append_rows(self._pred_path, self._PRED_COLS, rows)
        logger.info(f"[Monitor] logged {len(rows)} forecast steps for run_id={run_id!r}")

    def fill_actuals(self, actuals: pd.Series, forecast_ts: str, horizon_step_offset: int = 1) -> None:
        """Backfill actual values for a given forecast_ts.

        Args:
            actuals: actual observed values (length == horizon)
            forecast_ts: must match the forecast_ts used in log_forecast
            horizon_step_offset: first step index (usually 1)
        """
        rows = []
        for i, val in enumerate(actuals):
            rows.append({
                "forecast_ts": forecast_ts,
                "horizon_step": horizon_step_offset + i,
                "y_true": float(val),
            })
        self._append_rows(self._act_path, self._ACT_COLS, rows)
        logger.info(f"[Monitor] filled {len(rows)} actuals for forecast_ts={forecast_ts!r}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_rolling_metrics(self) -> dict[str, float]:
        """Join predictions and actuals, compute MAE/RMSE/MAPE on last `window` matched rows."""
        pred_df = self._read_csv(self._pred_path, self._PRED_COLS)
        act_df = self._read_csv(self._act_path, self._ACT_COLS)

        if pred_df.empty or act_df.empty:
            return {}

        pred_df["horizon_step"] = pd.to_numeric(pred_df["horizon_step"], errors="coerce")
        act_df["horizon_step"] = pd.to_numeric(act_df["horizon_step"], errors="coerce")
        merged = pred_df.merge(act_df, on=["forecast_ts", "horizon_step"], how="inner")

        if merged.empty:
            return {}

        merged["yhat"] = pd.to_numeric(merged["yhat"], errors="coerce")
        merged["y_true"] = pd.to_numeric(merged["y_true"], errors="coerce")
        merged = merged.dropna(subset=["yhat", "y_true"])
        tail = merged.tail(self.window)
        if tail.empty:
            return {}

        return {
            "mae": mae(tail["y_true"].values, tail["yhat"].values),
            "rmse": rmse(tail["y_true"].values, tail["yhat"].values),
            "mape": mape(tail["y_true"].values, tail["yhat"].values),
            "n_samples": len(tail),
        }

    def snapshot_metrics(self, run_id: str) -> dict[str, float]:
        """Compute rolling metrics and append a row to metrics_history.csv."""
        metrics = self.compute_rolling_metrics()
        if not metrics:
            logger.warning("[Monitor] no matched predictions/actuals to snapshot")
            return {}
        row = {
            "snapshot_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "run_id": run_id,
            "window": metrics.get("n_samples", self.window),
            "mae": metrics.get("mae", ""),
            "rmse": metrics.get("rmse", ""),
            "mape": metrics.get("mape", ""),
        }
        self._append_rows(self._metrics_path, self._METRICS_COLS, [row])
        return metrics

    def check_degradation(
        self,
        baseline_metrics: dict[str, float],
        threshold_ratio: float = 0.2,
    ) -> list[str]:
        """Compare rolling metrics to a baseline; return alert messages for degraded metrics.

        Args:
            baseline_metrics: reference values, e.g. {"mae": 3.2, "rmse": 4.5}
            threshold_ratio: alert if current > baseline * (1 + threshold_ratio)
        """
        rolling = self.compute_rolling_metrics()
        if not rolling:
            return ["[Monitor] insufficient data for degradation check"]

        alerts: list[str] = []
        for metric, baseline_val in baseline_metrics.items():
            current_val = rolling.get(metric)
            if current_val is None:
                continue
            if baseline_val <= 0:
                continue
            ratio = (current_val - baseline_val) / abs(baseline_val)
            if ratio > threshold_ratio:
                alerts.append(
                    f"[Monitor] {metric} degraded: "
                    f"current={current_val:.4f}, baseline={baseline_val:.4f}, "
                    f"change=+{ratio:.1%}"
                )
        if alerts:
            for alert in alerts:
                logger.warning(alert)
        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_headers(self) -> None:
        for path, cols in [
            (self._pred_path, self._PRED_COLS),
            (self._act_path, self._ACT_COLS),
            (self._metrics_path, self._METRICS_COLS),
        ]:
            if not path.exists():
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=cols)
                    writer.writeheader()

    def _append_rows(self, path: Path, cols: list[str], rows: list[dict]) -> None:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writerows(rows)

    def _read_csv(self, path: Path, cols: list[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=cols)
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            return pd.DataFrame(columns=cols)
