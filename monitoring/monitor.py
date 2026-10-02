"""监控闭环：预测日志写入、实际值回填、滚动指标计算与快照。"""
from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from config import AppConfig
from evaluation.metrics import POINT_METRICS
from utils.log_util import logger


def _utc_now_iso() -> str:
    """当前 UTC 时间的紧凑 ISO 字符串（Z 后缀）。"""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fmt_ts(value: object) -> str:
    """时间值写入字符串化：pd.Timestamp 用 isoformat，其余 str()，空值转空串。"""
    if value is None or value is pd.NaT or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = value.isoformat() if isinstance(value, pd.Timestamp) else str(value)
    return "" if text.strip().lower() in ("", "nan", "nat", "none") else text


def _canon_ts(value: object) -> str:
    """时间值比较规范化：空转 ""，可解析的统一转 UTC 去时区后 isoformat。

    兼容 "2026-05-04T00:00:00Z"（tz-aware）与 "2026-05-04 00:00:00"（naive）
    等书写差异——只统一到同一 epoch 时刻比较，不保留时区书写；
    用于 join 键与回填幂等键的统一比较。
    """
    text = _fmt_ts(value)
    if text == "":
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    if parsed.tz is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed.isoformat()


class ModelMonitor:
    """
    记录线上/离线预测质量，用于发现模型退化。

    monitor_dir/setting/ 下的文件约定：
        predictions_log.csv   — 每个预测步一行；多水平列 yhat_lower_80 等
        actuals_log.csv       — 后续回填的真实值
        metrics_history.csv   — 滚动指标快照
    """

    _PRED_BASE_COLS = ["run_id", "forecast_ts", "target_ts", "horizon_step", "yhat", "yhat_lower", "yhat_upper"]
    _ACT_BASE_COLS = ["forecast_ts", "target_ts", "horizon_step", "y_true"]
    _METRICS_BASE_COLS = ["snapshot_ts", "run_id", "window"]

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
        # 列状态每实例独立：多水平/新指标列只影响当前实例，不泄漏到其他 monitor
        self._pred_cols = list(self._PRED_BASE_COLS)
        self._act_cols = list(self._ACT_BASE_COLS)
        self._metrics_cols = list(self._METRICS_BASE_COLS)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_headers()

    @property
    def paths(self) -> dict[str, Path]:
        """三个监控日志的落盘路径（predictions / actuals / metrics）。"""
        return {
            "predictions": self._pred_path,
            "actuals": self._act_path,
            "metrics": self._metrics_path,
        }

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
        target_ts: Iterable[object] | None = None,
    ) -> None:
        """将一次预测结果追加写入 predictions_log.csv。

        target_ts 为逐预测步的目标时间戳（与 yhat 等长，可选）；双方日志都
        携带时滚动指标按 (forecast_ts, target_ts, horizon_step) 三键匹配，
        把预测点定位到目标时间而非仅发起时刻。旧表头缺列时写入自动补齐。
        """
        for name, bound in (("yhat_lower", yhat_lower), ("yhat_upper", yhat_upper)):
            if bound is not None and len(bound) != len(yhat):
                raise ValueError(f"{name} length {len(bound)} != yhat length {len(yhat)}")
        ts = forecast_ts or _utc_now_iso()
        targets = self._normalize_targets(target_ts, len(yhat))
        lowers = list(yhat_lower) if yhat_lower is not None else [""] * len(yhat)
        uppers = list(yhat_upper) if yhat_upper is not None else [""] * len(yhat)
        rows = [
            {
                "run_id": run_id,
                "forecast_ts": ts,
                "target_ts": target,
                "horizon_step": step,
                "yhat": float(val),
                "yhat_lower": lo if lo == "" else float(lo),
                "yhat_upper": up if up == "" else float(up),
            }
            for step, (val, target, lo, up) in enumerate(zip(yhat, targets, lowers, uppers), start=1)
        ]
        self._append_rows(self._pred_path, self._pred_cols, rows)
        logger.info(f"[Monitor] logged {len(rows)} forecast steps for run_id={run_id!r}")

    def log_forecast_levels(
        self,
        run_id: str,
        yhat: pd.Series,
        levels: list[float],
        bounds: dict[str, pd.Series],
        forecast_ts: str | None = None,
        target_ts: Iterable[object] | None = None,
    ) -> None:
        """多水平预测写入：bounds 键为 yhat_lower_80/yhat_upper_95 式带后缀列。

        列名由 forecasting.intervals.interval_bound_columns 协议产生；
        旧表头在写入时自动补列，单水平旧列保留空值。
        """
        from forecasting.intervals import interval_bound_columns

        multi = len(levels) > 1
        expected: list[str] = []
        for level in levels:
            lower_col, upper_col = interval_bound_columns(level, multi=multi)
            expected.extend([lower_col, upper_col])
        for col in expected:
            if col not in bounds:
                raise ValueError(f"bounds missing required column {col!r}")
        for col in expected:
            if col not in self._pred_cols:
                self._pred_cols.append(col)
        ts = forecast_ts or _utc_now_iso()
        targets = self._normalize_targets(target_ts, len(yhat))
        rows = []
        for step, val in enumerate(yhat, start=1):
            row: dict[str, object] = {
                "run_id": run_id,
                "forecast_ts": ts,
                "target_ts": targets[step - 1],
                "horizon_step": step,
                "yhat": float(val),
            }
            for col in expected:
                row[col] = float(bounds[col].iloc[step - 1])
            rows.append(row)
        self._append_rows(self._pred_path, self._pred_cols, rows)
        logger.info(f"[Monitor] logged {len(rows)} multi-level forecast steps for run_id={run_id!r}")

    def fill_actuals(
        self,
        actuals: pd.Series,
        forecast_ts: str,
        horizon_step_offset: int = 1,
        target_ts: Iterable[object] | None = None,
    ) -> None:
        """按 forecast_ts 回填真实值；已存在的相同键显式 RAISE。

        Args:
            actuals: 实际观测值，长度通常等于 horizon。
            forecast_ts: 必须与 log_forecast 中写入的 forecast_ts 匹配。
            horizon_step_offset: 起始预测步编号，默认从 1 开始。
            target_ts: 逐观测的目标时间戳（与 actuals 等长，可选）。
        """
        targets = self._normalize_targets(target_ts, len(actuals))
        rows = []
        for i, val in enumerate(actuals):
            rows.append({
                "forecast_ts": forecast_ts,
                "target_ts": targets[i],
                "horizon_step": horizon_step_offset + i,
                "y_true": float(val),
            })
        self._reject_duplicate_actuals(rows)
        self._append_rows(self._act_path, self._act_cols, rows)
        logger.info(f"[Monitor] filled {len(rows)} actuals for forecast_ts={forecast_ts!r}")

    def fill_actuals_frame(
        self,
        actuals_df: pd.DataFrame,
        *,
        actual_col: str = "y_true",
        forecast_ts: str | None = None,
        forecast_ts_col: str = "forecast_ts",
        horizon_step_col: str = "horizon_step",
        target_ts_col: str = "target_ts",
    ) -> None:
        """从结构化表回填真实值，适合 CLI 从 CSV 批量导入。

        输入表可逐行提供 forecast_ts/target_ts/horizon_step；缺 forecast_ts
        列时必须通过参数提供单个 forecast_ts。已存在的相同键显式 RAISE，
        防止重复回填稀释滚动指标。
        """
        if actual_col not in actuals_df.columns:
            raise ValueError(f"actual_col '{actual_col}' not found in actuals data")
        has_forecast_ts_col = forecast_ts_col in actuals_df.columns
        if not has_forecast_ts_col and forecast_ts is None:
            raise ValueError("forecast_ts is required when actuals data has no forecast_ts column")
        has_step_col = horizon_step_col in actuals_df.columns
        has_target_col = target_ts_col in actuals_df.columns
        n = len(actuals_df)
        steps = [int(v) for v in actuals_df[horizon_step_col]] if has_step_col else list(range(1, n + 1))
        fts = [_fmt_ts(v) for v in actuals_df[forecast_ts_col]] if has_forecast_ts_col else [str(forecast_ts)] * n
        tts = [_fmt_ts(v) for v in actuals_df[target_ts_col]] if has_target_col else [""] * n
        rows = [
            {"forecast_ts": f, "target_ts": t, "horizon_step": s, "y_true": float(v)}
            for f, t, s, v in zip(fts, tts, steps, actuals_df[actual_col])
        ]
        self._reject_duplicate_actuals(rows)
        self._append_rows(self._act_path, self._act_cols, rows)
        logger.info(f"[Monitor] filled {len(rows)} actual rows from dataframe")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_rolling_metrics(self) -> dict[str, float]:
        """合并预测与真实值，并在最近 window 个匹配样本上计算点误差与逐水平覆盖率。

        点指标遍历 evaluation.metrics.POINT_METRICS 注册表派生（requires_train
        的 mase/rmsse 依赖训练窗，监控场景跳过）。覆盖率键：单水平
        interval_coverage；多水平 interval_coverage_80 式带后缀。
        无区间列或区间值缺失时不产出对应键（不伪造覆盖率）。

        按记录匹配：双方 target_ts 非空用三键；一方为空才降级双键，
        多候选歧义显式失败。时间规范化容忍书写差异，混用新旧日志不丢配对。
        """
        pred_df = self._read_csv(self._pred_path, self._pred_cols)
        act_df = self._read_csv(self._act_path, self._act_cols)

        if pred_df.empty or act_df.empty:
            return {}

        pred_df["horizon_step"] = pd.to_numeric(pred_df["horizon_step"], errors="coerce")
        act_df["horizon_step"] = pd.to_numeric(act_df["horizon_step"], errors="coerce")
        # forecast_ts/target_ts 统一规范化后比较，容忍书写格式差异（Z 后缀/空格分隔）
        pred_df["forecast_ts"] = pred_df["forecast_ts"].map(_canon_ts)
        act_df["forecast_ts"] = act_df["forecast_ts"].map(_canon_ts)
        pred_df["target_ts"] = pred_df["target_ts"].map(_canon_ts)
        act_df["target_ts"] = act_df["target_ts"].map(_canon_ts)
        pred_df["_prediction_row"] = range(len(pred_df))
        act_df["_actual_row"] = range(len(act_df))
        merged = pred_df.merge(act_df, on=["forecast_ts", "horizon_step"], how="inner", suffixes=("_pred", "_actual"))
        pred_target, actual_target = merged["target_ts_pred"], merged["target_ts_actual"]
        # 按记录降级：只有缺少目标时间的一方才允许双键；双方非空却不同绝不误配。
        merged = merged.loc[pred_target.eq(actual_target) | pred_target.eq("") | actual_target.eq("")]
        if merged["_prediction_row"].duplicated().any() or merged["_actual_row"].duplicated().any():
            raise ValueError("ambiguous monitor match: supply unique forecast/target/step keys")
        merged = merged.sort_values("_prediction_row")

        if merged.empty:
            return {}

        merged["yhat"] = pd.to_numeric(merged["yhat"], errors="coerce")
        merged["y_true"] = pd.to_numeric(merged["y_true"], errors="coerce")
        merged = merged.dropna(subset=["yhat", "y_true"])
        tail = merged.tail(self.window)
        if tail.empty:
            return {}

        y_true = tail["y_true"].to_numpy(dtype=float)
        y_hat = tail["yhat"].to_numpy(dtype=float)
        out: dict[str, float] = {}
        for name, spec in POINT_METRICS.items():
            if spec.requires_train:
                continue
            out[name] = float(spec.func(y_true, y_hat))
        out["n_samples"] = len(tail)
        for lower_col, upper_col, key in self._iter_coverage_columns(merged.columns):
            lower = pd.to_numeric(tail[lower_col], errors="coerce")
            upper = pd.to_numeric(tail[upper_col], errors="coerce")
            valid = lower.notna() & upper.notna()
            if not valid.any():
                continue
            hits = ((tail["y_true"][valid] >= lower[valid]) & (tail["y_true"][valid] <= upper[valid]))
            out[key] = float(hits.mean())
        return out

    @staticmethod
    def _iter_coverage_columns(columns: Iterable[str]) -> Iterator[tuple[str, str, str]]:
        """区间列 → 覆盖率键：yhat_lower[_suffix] ↔ yhat_upper[_suffix]。"""
        cols = set(columns)
        for col in columns:
            if col == "yhat_lower" and "yhat_upper" in cols:
                yield col, "yhat_upper", "interval_coverage"
            elif col.startswith("yhat_lower_"):
                suffix = col[len("yhat_lower_"):]
                upper = f"yhat_upper_{suffix}"
                if upper in cols:
                    yield col, upper, f"interval_coverage_{suffix}"

    def snapshot_metrics(self, run_id: str) -> dict[str, float]:
        """计算滚动指标，并追加一条快照到 metrics_history.csv。"""
        metrics = self.compute_rolling_metrics()
        if not metrics:
            logger.warning("[Monitor] no matched predictions/actuals to snapshot")
            return {}
        row: dict[str, object] = {
            "snapshot_ts": _utc_now_iso(),
            "run_id": run_id,
            "window": metrics.get("n_samples", self.window),
        }
        for key in sorted(k for k in metrics if k != "n_samples"):
            value = metrics[key]
            # 非有限值（如 n<2 时的 r2=NaN）写空串，避免快照表出现 nan 字面量
            row[key] = value if (isinstance(value, int) or (isinstance(value, float) and isfinite(value))) else ""
            if key not in self._metrics_cols:
                self._metrics_cols.append(key)
        self._append_rows(self._metrics_path, self._metrics_cols, [row])
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_targets(target_ts: Iterable[object] | None, n: int) -> list[str]:
        """target_ts 统一为长度 n 的字符串列表；None 视为未提供（全空串）。"""
        if target_ts is None:
            return [""] * n
        values = list(target_ts)
        if len(values) != n:
            raise ValueError(f"target_ts length {len(values)} != expected length {n}")
        return [_fmt_ts(v) for v in values]

    def _reject_duplicate_actuals(self, rows: list[dict]) -> None:
        """批内与批间统一判重；缺目标时间的旧键不得和新三键歧义重叠。"""
        existing = self._read_csv(self._act_path, self._act_cols)
        keys: dict[tuple[str, float], set[str]] = {}
        for row in [*existing.to_dict("records"), *rows]:
            key = (_canon_ts(row["forecast_ts"]), float(row["horizon_step"]))
            target = _canon_ts(row.get("target_ts", ""))
            targets = keys.setdefault(key, set())
            if target in targets or (targets and (not target or "" in targets)):
                raise ValueError("actuals already filled: duplicate or ambiguous legacy/new key")
            targets.add(target)

    def _ensure_headers(self) -> None:
        """三个日志文件表头保障：不存在则建，存在但缺新列则原位补空列。

        旧版文件（如无 target_ts 列）在 monitor 接管目录时即完成迁移，
        旧行新列置空；不迁移会让按表头定位的 reader 读不到新列数据。
        """
        for path, cols in [
            (self._pred_path, self._pred_cols),
            (self._act_path, self._act_cols),
            (self._metrics_path, self._metrics_cols),
        ]:
            if not path.exists():
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=cols).writeheader()
                continue
            with open(path, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            missing = [c for c in cols if c not in header]
            if missing:
                self._add_columns_to_file(path, missing)
                logger.info(f"[Monitor] migrated {path.name} header on init: +{missing}")

    def _append_rows(self, path: Path, cols: list[str], rows: list[dict]) -> None:
        """追加写入；以文件实际表头顺序为 fieldnames，缺列时先原位补空列。

        旧文件补列后新列在表头末尾，与内存列序不同；DictWriter 按
        fieldnames 排列数据，必须跟随文件物理顺序，否则数据串列。
        """
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            missing = [c for c in cols if c not in header]
            if missing:
                self._add_columns_to_file(path, missing)
                header = header + missing
                logger.info(f"[Monitor] migrated {path.name} header: +{missing}")
        else:
            header = list(cols)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writerows(rows)

    @staticmethod
    def _add_columns_to_file(path: Path, new_cols: list[str]) -> None:
        """给已有 CSV 原位补空列（重写文件，旧行新列置空）。"""
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in new_cols:
            if col not in existing.columns:
                existing[col] = ""
        existing.to_csv(path, index=False)

    def _read_csv(self, path: Path, cols: list[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=cols)
        try:
            return pd.read_csv(path, dtype=str)
        except Exception as exc:
            # 损坏日志降级为空表但不静默：保留 warning 供排查
            logger.warning(f"[Monitor] failed to read {path}: {exc!r}; treating as empty")
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
    # 监控数据保存路径；data_name 解析与校验统一复用 artifacts.paths 的唯一实现
    from artifacts.paths import build_experiment_path, resolve_data_name

    data_name = resolve_data_name(cfg)
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
        "monitor_predictions_path": str(monitor.paths["predictions"]),
        "monitor_actuals_path": str(monitor.paths["actuals"]),
        "monitor_metrics_path": str(monitor.paths["metrics"]),
        "metrics": metrics,
    }
