from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import AppConfig
from .metrics import mae, rmse, mape
from utils.log_util import logger


class ModelMonitor:
    """
    记录线上/离线预测质量，用于发现模型退化。

    monitor_dir/setting/ 下的文件约定：
        predictions_log.csv   — 每个预测步一行
        actuals_log.csv       — 后续回填的真实值
        metrics_history.csv   — 滚动指标快照
    """

    _PRED_COLS = ["run_id", "forecast_ts", "horizon_step", "yhat", "yhat_lower", "yhat_upper"]
    _ACT_COLS = ["forecast_ts", "horizon_step", "y_true"]
    _METRICS_COLS = ["snapshot_ts", "run_id", "window", "mae", "rmse", "mape"]

    def __init__(self, monitor_dir: str | Path, setting: str | Path | None, window: int = 30):
        """
        Args:
            monitor_dir: 监控文件根目录。
            setting: 子目录名，通常与 model-data-strategy setting 一致。
            window: 计算滚动指标时使用的最近匹配样本数。
        """
        if window <= 0:
            raise ValueError("window must be > 0")
        self.monitor_dir = Path(monitor_dir) if setting is None else Path(monitor_dir) / setting
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
        """将一次预测结果追加写入 predictions_log.csv。"""
        ts = forecast_ts or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
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
        """按 forecast_ts 回填真实值。

        Args:
            actuals: 实际观测值，长度通常等于 horizon。
            forecast_ts: 必须与 log_forecast 中写入的 forecast_ts 匹配。
            horizon_step_offset: 起始预测步编号，默认从 1 开始。
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

    def fill_actuals_frame(
        self,
        actuals_df: pd.DataFrame,
        *,
        actual_col: str = "y_true",
        forecast_ts: str | None = None,
        forecast_ts_col: str = "forecast_ts",
        horizon_step_col: str = "horizon_step",
    ) -> None:
        """从结构化表回填真实值，适合 CLI 从 CSV 批量导入。

        输入表可以逐行提供 forecast_ts/horizon_step；如果没有 forecast_ts
        列，则必须通过参数提供单个 forecast_ts。
        """
        if actual_col not in actuals_df.columns:
            raise ValueError(f"actual_col '{actual_col}' not found in actuals data")
        has_forecast_ts_col = forecast_ts_col in actuals_df.columns
        if not has_forecast_ts_col and forecast_ts is None:
            raise ValueError("forecast_ts is required when actuals data has no forecast_ts column")

        rows = []
        for idx, row in actuals_df.reset_index(drop=True).iterrows():
            step = int(row[horizon_step_col]) if horizon_step_col in actuals_df.columns else int(idx + 1)
            rows.append(
                {
                    "forecast_ts": str(row[forecast_ts_col]) if has_forecast_ts_col else str(forecast_ts),
                    "horizon_step": step,
                    "y_true": float(row[actual_col]),
                }
            )
        self._append_rows(self._act_path, self._ACT_COLS, rows)
        logger.info(f"[Monitor] filled {len(rows)} actual rows from dataframe")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_rolling_metrics(self) -> dict[str, float]:
        """合并预测与真实值，并在最近 window 个匹配样本上计算 MAE/RMSE/MAPE。"""
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
        """计算滚动指标，并追加一条快照到 metrics_history.csv。"""
        metrics = self.compute_rolling_metrics()
        if not metrics:
            logger.warning("[Monitor] no matched predictions/actuals to snapshot")
            return {}
        row = {
            "snapshot_ts": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
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
        """将当前滚动指标与基线比较，返回退化告警信息。

        Args:
            baseline_metrics: 参考指标，例如 {"mae": 3.2, "rmse": 4.5}。
            threshold_ratio: 当前指标超过 baseline * (1 + threshold_ratio) 时告警。
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


def run_monitor_actuals_backfill(cfg: AppConfig) -> dict[str, Any] | None:
    """
    从 CSV 回填监控 actuals，并可选写入滚动指标快照。

    兼具 CLI early-return 判断：未配置 monitor_actuals_path 则返回 None；
    已配置时使用 AppConfig 中的 monitor 参数执行回填。
    """
    # 未配置 monitor_actuals_path 则返回 None
    if cfg.monitor_actuals_path is None:
        return None
    # 监控数据保存路径
    from app.results import build_experiment_path

    data_name = "demo_series" if cfg.data_path is None else Path(cfg.data_path).stem
    experiment_path = Path(cfg.monitor_actuals_experiment_path) if cfg.monitor_actuals_experiment_path else build_experiment_path(cfg)
    if experiment_path.is_absolute() or ".." in experiment_path.parts:
        raise ValueError("monitor_actuals_experiment_path must be a relative path under the data monitor directory")
    # 读取回填数据
    actuals_df = pd.read_csv(cfg.monitor_actuals_path)
    # 创建 Monitor
    monitor_root = Path(cfg.results_dir) / data_name / "monitor" / experiment_path
    monitor = ModelMonitor(monitor_dir=monitor_root, setting=None, window=cfg.monitor_window)
    # 真实值回填
    monitor.fill_actuals_frame(
        actuals_df,
        actual_col=cfg.monitor_actuals_value_col,
        forecast_ts=cfg.monitor_actuals_forecast_ts,
    )
    # 写入滚动指标
    metrics = (
        monitor.snapshot_metrics(run_id=cfg.monitor_actuals_run_id)
        if cfg.monitor_actuals_snapshot
        else monitor.compute_rolling_metrics()
    )

    return {
        "monitor_predictions_path": str(monitor._pred_path),
        "monitor_actuals_path": str(monitor._act_path),
        "monitor_metrics_path": str(monitor._metrics_path),
        "metrics": metrics,
    }
