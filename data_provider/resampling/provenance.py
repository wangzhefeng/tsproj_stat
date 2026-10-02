"""只读核验聚合来源，不触发缓存重建，不修改任何输入。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .service import _AUDIT_VERSION, _audit_digest, _file_digest


def inspect_aggregation_audit(data_path: str | Path, *, freq: str, time_col: str, target_col: str) -> dict:
    data = Path(data_path).resolve()
    path = data.with_name(data.name + ".aggregate.json")
    result: dict = {"status": "missing", "data_path": str(data), "audit_path": str(path),
                    "reason": "aggregation audit missing; upstream repair/as-of status unknown"}
    if not path.exists():
        return result
    try:
        content = path.read_bytes()
        result["audit_file_sha256"] = hashlib.sha256(content).hexdigest()
        audit = json.loads(content)
        if not isinstance(audit, dict) or audit.get("audit_version") != _AUDIT_VERSION:
            raise ValueError("unsupported audit version")
        if audit.get("audit_sha256") != _audit_digest(audit):
            raise ValueError("audit digest mismatch")
        for name in ("source_rows", "output_rows", "inserted_timestamp_count", "filled_value_count"):
            if type(audit.get(name)) is not int or audit[name] < 0:
                raise ValueError(f"invalid audit count: {name}")
        if audit.get("output_sha256") != _file_digest(data):
            raise ValueError("derived CSV digest mismatch")
        cfg = audit["config"]
        if (cfg["time_col"] != time_col or cfg["target_col"] != target_col
                or pd.tseries.frequencies.to_offset(cfg["target_freq"]) != pd.tseries.frequencies.to_offset(freq)):
            raise ValueError("audit column/frequency mismatch")
        if type(audit.get("fill_uses_future")) is not bool:
            raise ValueError("missing fill direction disclosure")
        source = Path(cfg["source_path"])
        if source.is_file() and _file_digest(source) != cfg["source_sha256"]:
            raise ValueError("upstream source digest mismatch")
        result.update(status="verified" if source.is_file() else "source_unavailable",
                      reason="audit/output/source digests match" if source.is_file() else "audit/output match; upstream source unavailable",
                      source_path=str(source), source_sha256=cfg["source_sha256"], output_sha256=audit["output_sha256"],
                      source_freq=cfg["source_freq"], method=cfg["method"], fill_method=cfg["fill_method"],
                      fill_uses_future=audit["fill_uses_future"], fill_direction_note=audit.get("fill_direction_note", ""),
                      filled_value_count=audit["filled_value_count"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="invalid", reason=str(exc))
    return result
