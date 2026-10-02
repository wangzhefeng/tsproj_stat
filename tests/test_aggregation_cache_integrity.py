"""缓存验证包含文件内容，发布中断不得留下可复用的错误结果。"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from data_provider.resampling.service import aggregate_csv


def _aggregate(kwargs):
    return aggregate_csv(**kwargs)


@pytest.fixture
def aggregation(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("ds,y\n2026-01-01,1\n2026-01-02,3\n")
    return dict(source_path=source, output_path=tmp_path / "output.csv", time_col="ds",
                target_col="y", source_freq="D", target_freq="2D")


@pytest.mark.parametrize("damage", ["csv", "source", "audit_json", "audit_shape", "audit_fields", "audit_count", "legacy"])
def test_cache_damage_requires_rebuild(aggregation, damage):
    result = aggregate_csv(**aggregation)
    source = aggregation["source_path"]
    expected = [2.]
    if damage == "csv":
        result.data_path.write_text("ds,y\n2026-01-01,999\n")
    elif damage == "source":
        stat = source.stat()
        source.write_text(source.read_text().replace(",3", ",9"))
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        expected = [5.]
    elif damage == "audit_json":
        result.audit_path.write_text("{")
    elif damage == "audit_shape":
        result.audit_path.write_text("[]")
    else:
        audit = json.loads(result.audit_path.read_text())
        if damage == "audit_fields":
            del audit["output_rows"]
        elif damage == "audit_count":
            audit["output_rows"] = 999
        else:
            audit.pop("audit_version", None)
        result.audit_path.write_text(json.dumps(audit))
    rebuilt = aggregate_csv(**aggregation)
    assert rebuilt.regenerated
    assert pd.read_csv(rebuilt.data_path)["y"].tolist() == expected
    assert not aggregate_csv(**aggregation).regenerated


def test_same_destination_multiprocess_publication(aggregation):
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(_aggregate, [aggregation] * 6))
    assert sum(r.regenerated for r in results) == 1
    result = results[-1]
    audit = json.loads(result.audit_path.read_text())
    assert audit["output_sha256"] == hashlib.sha256(result.data_path.read_bytes()).hexdigest()
    assert pd.read_csv(result.data_path)["y"].tolist() == [2.]


def test_interrupted_pair_publish_is_not_reused(aggregation, monkeypatch):
    result = aggregate_csv(**aggregation)
    aggregation["source_path"].write_text("ds,y\n2026-01-01,5\n2026-01-02,9\n")
    real_replace = os.replace
    def fail_audit(src, dst):
        if Path(dst) == result.audit_path:
            raise OSError("injected audit publish failure")
        return real_replace(src, dst)
    with monkeypatch.context() as patcher:
        patcher.setattr(os, "replace", fail_audit)
        with pytest.raises(OSError, match="injected"):
            aggregate_csv(**aggregation)
    repaired = aggregate_csv(**aggregation)
    assert repaired.regenerated
    assert pd.read_csv(repaired.data_path)["y"].tolist() == [7.]
    assert not aggregate_csv(**aggregation).regenerated
