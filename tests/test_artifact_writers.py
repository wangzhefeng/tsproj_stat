import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from artifacts.writers import dataframe_to_csv, write_json


def test_json_normalizes_scientific_values_and_rejects_unknown(tmp_path):
    path = tmp_path / "out.json"
    write_json(path, {"n": np.int64(7), "values": np.array([1., np.nan, np.inf]),
                      "time": pd.Timestamp("2024-01-01"), "missing": pd.NaT, "path": Path("x")})
    def reject(value):
        raise AssertionError(value)
    result = json.loads(path.read_text(), parse_constant=reject)
    assert result == {"n": 7, "values": [1., None, None], "time": "2024-01-01T00:00:00",
                      "missing": None, "path": "x"}
    previous = path.read_bytes()
    with pytest.raises(TypeError):
        write_json(path, {"bad": object()})
    assert path.read_bytes() == previous


def test_csv_failure_preserves_previous_target(tmp_path, monkeypatch):
    path = tmp_path / "out.csv"
    dataframe_to_csv(path, pd.DataFrame({"x": [1, 2]}))
    original = path.read_bytes()
    def fail(self, target, **kwargs):
        Path(target).write_text("partial")
        raise OSError("controlled disk failure")
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="controlled"):
        dataframe_to_csv(path, pd.DataFrame({"x": [3]}))
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.csv"]


def test_pickle_and_replace_failures_preserve_target(tmp_path, monkeypatch):
    from artifacts import writers
    path = tmp_path / "model.pkl"
    writers.write_pickle(path, [1, 2, 3])
    before = path.read_bytes()
    with pytest.raises(Exception):
        writers.write_pickle(path, lambda: None)
    assert path.read_bytes() == before
    def fail(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(writers.os, "replace", fail)
    with pytest.raises(OSError, match="replace"):
        writers.write_pickle(path, [4, 5])
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["model.pkl"]


def test_concurrent_json_versions_are_complete(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "out.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: write_json(path, {"n": n, "data": [n] * 1000}), range(16)))
    data = json.loads(path.read_text())
    assert data["data"] == [data["n"]] * 1000
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]
