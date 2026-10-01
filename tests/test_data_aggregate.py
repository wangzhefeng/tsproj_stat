import json

import pandas as pd
import pytest

from data_provider.resampling.service import aggregate_csv


def _write_source(tmp_path, days=3, freq="h"):
    times = pd.date_range("2026-01-01", periods=24 * days, freq=freq)
    df = pd.DataFrame({"time": times, "value": [float(i % 24) for i in range(len(times))]})
    source = tmp_path / "source.csv"
    df.to_csv(source, index=False)
    return source


def test_aggregate_csv_generates_derived_csv_and_audit(tmp_path):
    source = _write_source(tmp_path)
    output = tmp_path / "derived" / "daily.csv"

    result = aggregate_csv(
        source_path=source,
        time_col="time",
        target_col="value",
        source_freq="h",
        target_freq="D",
        method="mean",
        fill_method="none",
        output_path=output,
    )

    assert result.regenerated is True
    assert result.output_rows == 3
    assert output.exists()
    derived = pd.read_csv(output)
    # 每日 24 个点取均值：0..23 的均值为 11.5
    assert derived["value"].tolist() == [11.5, 11.5, 11.5]

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["config"]["method"] == "mean"
    assert audit["config"]["source_freq"] == "h"
    assert audit["config"]["target_freq"] == "D"
    assert audit["source_rows"] == 72
    assert audit["output_rows"] == 3


def test_aggregate_csv_reuse_and_config_mismatch_regenerate(tmp_path):
    source = _write_source(tmp_path)
    output = tmp_path / "derived" / "daily.csv"
    kwargs = dict(
        source_path=source,
        time_col="time",
        target_col="value",
        source_freq="h",
        target_freq="D",
        method="mean",
        fill_method="none",
        output_path=output,
    )

    first = aggregate_csv(**kwargs)
    second = aggregate_csv(**kwargs)
    assert first.regenerated is True
    assert second.regenerated is False

    # 任一聚合参数变化都必须重算（审计不匹配不得静默复用）
    third = aggregate_csv(**{**kwargs, "method": "max"})
    assert third.regenerated is True
    derived = pd.read_csv(output)
    assert derived["value"].tolist() == [23.0, 23.0, 23.0]


def test_aggregate_csv_fill_disclosure_in_audit(tmp_path):
    """T16：审计 JSON 必须披露填充方向（双向填充不是 as-of 操作）。"""
    source = _write_source(tmp_path)

    for fill_method, expected_uses_future in [("none", False), ("linear", True), ("seasonal_slot", True)]:
        output = tmp_path / f"derived_{fill_method}.csv"
        result = aggregate_csv(
            source_path=source,
            time_col="time",
            target_col="value",
            source_freq="h",
            target_freq="D",
            method="mean",
            fill_method=fill_method,
            output_path=output,
        )
        audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
        assert audit["fill_uses_future"] is expected_uses_future
        assert isinstance(audit["fill_direction_note"], str) and audit["fill_direction_note"]


def test_aggregate_csv_rejects_invalid_method_and_self_overwrite(tmp_path):
    source = _write_source(tmp_path)

    with pytest.raises(ValueError, match="aggregation_method"):
        aggregate_csv(
            source_path=source,
            time_col="time",
            target_col="value",
            source_freq="h",
            target_freq="D",
            method="p95",
            output_path=tmp_path / "out.csv",
        )

    with pytest.raises(ValueError, match="must not overwrite"):
        aggregate_csv(
            source_path=source,
            time_col="time",
            target_col="value",
            source_freq="h",
            target_freq="D",
            method="mean",
            output_path=source,
        )
