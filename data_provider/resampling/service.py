"""文件级频率聚合：CSV 读取、派生文件落盘、审计 JSON 与缓存复用。

内存计算归 core.py；本模块负责文件 IO 和审计披露；
AppConfig 适配由 pipeline.data_preparation 承担。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .core import aggregate_frame, validate_aggregation_options

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
        "time_col": time_col,
        "target_col": target_col,
        "source_freq": source_freq,
        "target_freq": target_freq,
        "method": method,
        "fill_method": fill_method,
        "fill_weeks": int(fill_weeks),
    }


def _can_reuse(output_path: Path, audit_path: Path, expected_config: dict[str, Any]) -> bool:
    if not output_path.exists() or not audit_path.exists():
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return audit.get("config") == expected_config


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
    """把规则化后的单目标时间序列聚合到目标频率，并落盘审计信息。"""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Aggregation source file not found: {source}")
    validate_aggregation_options(method, fill_method, fill_weeks)

    destination = Path(output_path) if output_path else _default_output_path(source, target_freq, method)
    if destination.resolve() == source.resolve():
        raise ValueError("aggregation_output_path must not overwrite data_path")
    audit_path = destination.with_name(f"{destination.name}.aggregate.json")
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
    if _can_reuse(destination, audit_path, config):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        # 迁移旧独立脚本缓存：仅补齐审计披露，不重算或改写派生 CSV。
        if "fill_uses_future" not in audit or "fill_direction_note" not in audit:
            audit["fill_uses_future"], audit["fill_direction_note"] = _FILL_DISCLOSURE[fill_method]
            temp_audit = audit_path.with_name(f".{audit_path.name}.tmp")
            temp_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp_audit, audit_path)
        return AggregationResult(
            data_path=destination,
            audit_path=audit_path,
            regenerated=False,
            source_rows=int(audit["source_rows"]),
            output_rows=int(audit["output_rows"]),
            inserted_timestamp_count=int(audit["inserted_timestamp_count"]),
            filled_value_count=int(audit["filled_value_count"]),
        )

    computed = aggregate_frame(
        pd.read_csv(source), time_col=time_col, target_col=target_col,
        source_freq=source_freq, target_freq=target_freq, method=method,
        fill_method=fill_method, fill_weeks=fill_weeks,
    )
    aggregated = computed.frame
    source_rows = computed.source_rows
    inserted_count = computed.inserted_timestamp_count
    filled_count = computed.filled_value_count

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = destination.with_name(f".{destination.name}.tmp")
    temp_audit = audit_path.with_name(f".{audit_path.name}.tmp")
    aggregated.to_csv(temp_csv, index=False)
    audit = {
        "config": config,
        "source_rows": source_rows,
        "output_rows": len(aggregated),
        "inserted_timestamp_count": inserted_count,
        "filled_value_count": filled_count,
        "fill_uses_future": _FILL_DISCLOSURE[fill_method][0],
        "fill_direction_note": _FILL_DISCLOSURE[fill_method][1],
        "duplicate_timestamp_count": computed.duplicate_timestamp_count,
        "time_range_start": str(aggregated[time_col].iloc[0]),
        "time_range_end": str(aggregated[time_col].iloc[-1]),
    }
    temp_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_csv, destination)
    os.replace(temp_audit, audit_path)
    return AggregationResult(
        data_path=destination,
        audit_path=audit_path,
        regenerated=True,
        source_rows=source_rows,
        output_rows=len(aggregated),
        inserted_timestamp_count=inserted_count,
        filled_value_count=filled_count,
    )
