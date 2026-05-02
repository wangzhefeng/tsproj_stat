import pandas as pd
import pytest

from data_provider.data_transfer import validate_horizon, to_dataframe, to_univariate_series


def test_validate_horizon_rejects_non_positive():
    with pytest.raises(ValueError, match="horizon must be positive"):
        validate_horizon(0)


def test_to_univariate_series_uses_first_column():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

    series = to_univariate_series(frame)

    assert list(series) == [1, 2, 3]


def test_to_dataframe_wraps_series():
    series = pd.Series([1, 2, 3], name="y")

    frame = to_dataframe(series)

    assert list(frame.columns) == ["y"]
    assert frame["y"].tolist() == [1, 2, 3]
