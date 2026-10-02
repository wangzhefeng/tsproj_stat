"""文件级频率聚合：CSV 读取、派生文件落盘、审计 JSON 与缓存复用。

内存计算归 core.py；本模块负责文件 IO 和审计披露；
AppConfig 适配由 pipeline.data_preparation 承担。
"""
from __future__ import annotations

import json
import os
import fcntl
import hashlib
import tempfile
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .core import aggregate_frame, validate_aggregation_options

_AUDIT_VERSION = 2

# 填充方向披露（T16）：linear 的 limit_direction="both" 与 seasonal_slot 的 ±fill_weeks
# 双向窗口都会用未来观测回填过去的缺失，不是 as-of 操作；离线数据准备可接受，但必须披露。
_FILL_DISCLOSURE = {
    "none": (False, "no filling; as-of safe"),
    "linear": (True, "linear interpolation with limit_direction='both'; past gaps may use future observations (not as-of)"),
    "seasonal_slot": (True, "bidirectional ±fill_weeks same (weekday, minute-of-day) window; past gaps use future observations (not as-of)"),
}


@dataclass(frozen=True)
class AggregationResult:
    """一次聚合的产物定位与统计：派生 CSV/审计路径、是否重算、行数与补缺计数。"""

    data_path: Path
    audit_path: Path
    regenerated: bool
    source_rows: int
    output_rows: int
    inserted_timestamp_count: int
    filled_value_count: int


def _default_output_path(source_path: Path, target_freq: str, method: str) -> Path:
    safe_freq = "".join(char if char.isalnum() else "-" for char in target_freq).strip("-")
    return source_path.parent / "derived" / f"{source_path.stem}__{safe_freq}_{method}.csv"


def _audit_config(
    *,
    source_path: Path,
    time_col: str,
    target_col: str,
    source_freq: str,
    target_freq: str,
    method: str,
    fill_method: str,
    fill_weeks: int,
) -> dict[str, Any]:
    stat = source_path.stat()
    return {
        "source_path": str(source_path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_sha256": _file_digest(source_path),
        "time_col": time_col,
        "target_col": target_col,
        "source_freq": source_freq,
        "target_freq": target_freq,
        "method": method,
        "fill_method": fill_method,
        "fill_weeks": int(fill_weeks),
    }


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _audit_digest(audit: dict) -> str:
    payload = {key: value for key, value in audit.items() if key != "audit_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _read_cache(output_path: Path, audit_path: Path, expected_config: dict[str, Any]) -> dict | None:
    """版本、配置、计数和两份内容摘要一致才可信；旧审计一次性重建。"""
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict) or audit.get("audit_version") != _AUDIT_VERSION:
            return None
        counts = ("source_rows", "output_rows", "inserted_timestamp_count", "filled_value_count")
        if any(type(audit.get(key)) is not int or audit[key] < 0 for key in counts):
            return None
        if audit.get("config") != expected_config or audit.get("audit_sha256") != _audit_digest(audit):
            return None
        if audit.get("output_sha256") != _file_digest(output_path):
            return None
        return audit
    except (OSError, ValueError):
        return None


def _publish(frame: pd.DataFrame, destination: Path, audit_path: Path, audit: dict) -> None:
    """两个 replace 不是联合事务；审计最后发布，摘要识别中断残留。

    TemporaryDirectory 提供每次发布独有的同文件系统临时路径并自动清理。
    """
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as staging:
        csv = Path(staging) / "data.csv"
        sidecar = Path(staging) / "audit.json"
        frame.to_csv(csv, index=False)
        audit["output_sha256"] = _file_digest(csv)
        audit["audit_sha256"] = _audit_digest(audit)
        sidecar.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(csv, destination)
        os.replace(sidecar, audit_path)


def aggregate_csv(
    *,
    source_path: str | Path,
    time_col: str,
    target_col: str,
    source_freq: str,
    target_freq: str,
    method: str = "mean",
    fill_method: str = "none",
    fill_weeks: int = 4,
    output_path: str | Path | None = None,
) -> AggregationResult:
    """同目标进程锁覆盖读取、校验、计算、发布（macOS/Linux）。

    锁文件保留，不能 unlink：否则等待者和新进程会锁住不同 inode。
    """
    source = Path(source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Aggregation source file not found: {source}")
    validate_aggregation_options(method, fill_method, fill_weeks)

    destination = (Path(output_path) if output_path else _default_output_path(source, target_freq, method)).resolve()
    audit_path = destination.with_name(f"{destination.name}.aggregate.json")
    lock_path = destination.with_name(f".{destination.name}.lock")
    if source in {destination, audit_path, lock_path}:
        raise ValueError("aggregation_output_path must not overwrite data_path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _aggregate_locked(source, destination, audit_path, time_col, target_col,
                                     source_freq, target_freq, method, fill_method, fill_weeks)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _aggregate_locked(source: Path, destination: Path, audit_path: Path, time_col: str,
                      target_col: str, source_freq: str, target_freq: str, method: str,
                      fill_method: str, fill_weeks: int) -> AggregationResult:
    config = _audit_config(
        source_path=source,
        time_col=time_col,
        target_col=target_col,
        source_freq=source_freq,
        target_freq=target_freq,
        method=method,
        fill_method=fill_method,
        fill_weeks=fill_weeks,
    )
    audit = _read_cache(destination, audit_path, config)
    if audit is not None:
        return AggregationResult(
            data_path=destination,
            audit_path=audit_path,
            regenerated=False,
            source_rows=int(audit["source_rows"]),
            output_rows=int(audit["output_rows"]),
            inserted_timestamp_count=int(audit["inserted_timestamp_count"]),
            filled_value_count=int(audit["filled_value_count"]),
        )

    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != config["source_sha256"]:
        raise ValueError("Aggregation source changed before reading; retry with a stable source")
    computed = aggregate_frame(
        pd.read_csv(BytesIO(source_bytes)), time_col=time_col, target_col=target_col,
        source_freq=source_freq, target_freq=target_freq, method=method,
        fill_method=fill_method, fill_weeks=fill_weeks,
    )
    aggregated = computed.frame
    source_rows = computed.source_rows
    inserted_count = computed.inserted_timestamp_count
    filled_count = computed.filled_value_count

    if _file_digest(source) != config["source_sha256"]:
        raise ValueError("Aggregation source changed during computation; retry with a stable source")
    audit = {
        "audit_version": _AUDIT_VERSION,
        "config": config,
        "source_rows": source_rows,
        "output_rows": len(aggregated),
        "inserted_timestamp_count": inserted_count,
        "filled_value_count": filled_count,
        "fill_uses_future": _FILL_DISCLOSURE[fill_method][0],
        "fill_direction_note": _FILL_DISCLOSURE[fill_method][1],
        "duplicate_timestamp_count": computed.duplicate_timestamp_count,
        "duplicate_policy": "mean_at_source_timestamp",
        "source_grid_policy": "aligned_observations_only",
        "time_range_start": str(aggregated[time_col].iloc[0]),
        "time_range_end": str(aggregated[time_col].iloc[-1]),
    }
    _publish(aggregated, destination, audit_path, audit)
    return AggregationResult(
        data_path=destination,
        audit_path=audit_path,
        regenerated=True,
        source_rows=source_rows,
        output_rows=len(aggregated),
        inserted_timestamp_count=inserted_count,
        filled_value_count=filled_count,
    )
