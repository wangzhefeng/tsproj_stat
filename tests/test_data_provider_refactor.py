"""数据层通用边界与场景入口的可观察行为回归。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_provider.loading.loader import DataLoader


@pytest.mark.parametrize("method,expected", [
    ("mean", [2.0, 6.0]), ("max", [3.0, 7.0]),
    ("min", [1.0, 5.0]), ("sum", [4.0, 12.0]), ("median", [2.0, 6.0]),
])
def test_memory_aggregation_is_pure_and_preserves_methods(method, expected):
    from data_provider.resampling.core import aggregate_frame

    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=4, freq="h"),
                          "value": [1.0, 3.0, 5.0, 7.0], "unused": [0, 0, 0, 0]})
    original = frame.copy(deep=True)
    result = aggregate_frame(frame, time_col="time", target_col="value",
                             source_freq="h", target_freq="2h", method=method)
    assert result.frame["value"].tolist() == expected
    assert_frame_equal(frame, original)
    assert result.source_rows == 4
    assert result.inserted_timestamp_count == result.filled_value_count == 0


def test_memory_aggregation_gap_and_failure_contract():
    from data_provider.resampling.core import aggregate_frame

    frame = pd.DataFrame({"time": ["2026-01-01 00:00", "2026-01-01 02:00"], "value": [1.0, 5.0]})
    kwargs = dict(time_col="time", target_col="value", source_freq="h", target_freq="D")
    with pytest.raises(ValueError, match="missing source-frequency"):
        aggregate_frame(frame, **kwargs)
    result = aggregate_frame(frame, **kwargs, fill_method="linear")
    assert result.frame["value"].tolist() == [3.0]
    assert result.inserted_timestamp_count == result.filled_value_count == 1


@pytest.mark.parametrize("future", [False, True])
def test_csv_and_memory_share_cleaning_without_mutating_input(tmp_path, future):
    raw = pd.DataFrame({"ds": ["2026-01-03", "2026-01-01", "2026-01-02"],
                        "y": [5.0, 1.0, np.nan], "temp": [30.0, 10.0, np.inf]})
    original = raw.copy(deep=True)
    path = tmp_path / "input.csv"
    raw.to_csv(path, index=False)
    if future:
        a = DataLoader(None, future_exog_path=str(path), future_exog_time_col="ds")
        b = DataLoader(None, future_exog_frame=raw, future_exog_time_col="ds")
        outputs = [loader.load_future_exog(["temp"]) for loader in (a, b)]
        expected = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3), "temp": [10.0, np.nan, 30.0]})
    else:
        a = DataLoader(str(path), value_cols=["y", "temp"], max_missing_ratio=0.5)
        b = DataLoader(None, data_frame=raw, value_cols=["y", "temp"], max_missing_ratio=0.5)
        outputs = [loader.load_data() for loader in (a, b)]
        expected = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3),
                                 "y": [1.0, np.nan, 5.0], "temp": [10.0, np.nan, 30.0]})
        assert a.quality_report is not None and b.quality_report is not None
        assert a.quality_report.to_dict() == b.quality_report.to_dict()
    for output in outputs:
        assert output is not None
        assert_frame_equal(output, expected)
    assert_frame_equal(raw, original)


def test_aidc_entry_runs_from_other_cwd_and_reuses_audited_outputs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/aidc_power_month/prepare_data.py"
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    output_dir = tmp_path / "output"
    date_range = "fixture"
    times = pd.date_range("2026-01-01", periods=12, freq="5min")
    for route, offset in [("A", 0), ("B", 100)]:
        pd.DataFrame({"time": times, "value": np.arange(12, dtype=float) + offset}).to_csv(
            source_dir / f"{route}_Loads_5min_{date_range}.csv", index=False)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    command = [sys.executable, str(script), "--data-dir", str(source_dir),
               "--output-dir", str(output_dir), "--date-range", date_range]
    first = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert first.stdout.count("重新生成") == 6
    expected_files = set()
    for route, offset in [("A", 0), ("B", 100)]:
        for label, values in [("15min", [1.0, 4.0, 7.0, 10.0]), ("1hour", [5.5]), ("1day", [5.5])]:
            name = f"{route}_Loads_{label}_mean_{date_range}.csv"
            expected_files.update([name, name + ".aggregate.json"])
            assert pd.read_csv(output_dir / name)["value"].tolist() == [v + offset for v in values]
            audit = json.loads((output_dir / (name + ".aggregate.json")).read_text())
            assert audit["fill_uses_future"] is True
            assert audit["config"]["fill_method"] == "seasonal_slot"
    assert {p.name for p in output_dir.iterdir()} == expected_files
    before = {p.name: p.read_bytes() for p in output_dir.iterdir()}
    # 旧独立脚本的缓存没有填充方向披露；复用 CSV 时也应补齐审计。
    for audit_path in output_dir.glob("*.aggregate.json"):
        audit = json.loads(audit_path.read_text())
        del audit["fill_uses_future"]
        del audit["fill_direction_note"]
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    second = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("复用缓存") == 6
    for path in output_dir.iterdir():
        if path.suffix == ".json":
            assert json.loads(path.read_bytes()) == json.loads(before[path.name])
        else:
            assert path.read_bytes() == before[path.name]
