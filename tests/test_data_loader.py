from pathlib import Path

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
