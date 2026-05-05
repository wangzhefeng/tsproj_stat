from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_provider.data_loader import DataLoader


def test_data_loader_adds_fallback_time_column(tmp_path):
    csv_path = tmp_path / "series.csv"
    pd.DataFrame({"y": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)

    df = DataLoader(data_path=str(csv_path)).load_data()

    assert list(df.columns) == ["ds", "y"]
    assert str(df["ds"].iloc[0].date()) == "2000-01-01"


def test_data_loader_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        DataLoader(data_path=str(missing)).load_data()


def test_data_loader_split_history_future_requires_enough_samples():
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=5, freq="D"), "y": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError, match="history_size \\+ horizon"):
        DataLoader(data_path=None).split_history_future(df=df, history_size=4, horizon=2)


def test_data_loader_without_path_uses_demo_series():
    df = DataLoader(data_path=None).load_data()

    assert list(df.columns) == ["ds", "y"]
    assert len(df) == 200


def test_data_loader_normalizes_target_values_and_keeps_standard_columns(tmp_path):
    csv_path = tmp_path / "dirty_series.csv"
    pd.DataFrame(
        {
            "ds": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04"],
            "y": ["1.5", np.inf, "3.5", None],
            "extra": [10, 20, 30, 40],
        }
    ).to_csv(csv_path, index=False)

    df = DataLoader(data_path=str(csv_path)).load_data()

    assert list(df.columns) == ["ds", "y"]
    assert df["ds"].tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]))
    assert df["y"].tolist() == [3.5, 3.5, 1.5, 1.5]


def test_data_quality_report_tracks_cleaning_counts(tmp_path):
    csv_path = tmp_path / "dirty_series.csv"
    pd.DataFrame(
        {
            "ds": ["2024-01-01", "2024-01-03", "2024-01-04"],
            "y": [1.0, None, 4.0],
        }
    ).to_csv(csv_path, index=False)

    loader = DataLoader(data_path=str(csv_path), freq="D")
    loader.load_data()
    report = loader.quality_report.to_dict()

    assert report["raw_rows"] == 3
    assert report["clean_rows"] == 3
    assert report["interpolated_value_count"] >= 1
    assert report["inserted_timestamp_count"] == 1
    assert report["dropped_row_count"] == 0


def test_data_loader_requires_target_column(tmp_path):
    csv_path = tmp_path / "missing_target.csv"
    pd.DataFrame({"ds": ["2024-01-01", "2024-01-02"], "value": [1.0, 2.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="target_col 'y' not found"):
        DataLoader(data_path=str(csv_path)).load_data()
