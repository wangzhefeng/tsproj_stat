import pandas as pd
import pytest

from pipeline.windows import align_future_exog, forecast_timestamps


@pytest.mark.parametrize("origin,freq,expected", [
    ("2024-01-01", "D", ["2024-01-02", "2024-01-03"]),
    ("2024-01-15", "ME", ["2024-01-31", "2024-02-29"]),
    ("2024-01-31", "ME", ["2024-02-29", "2024-03-31"]),
    ("2024-01-01 00:00+08:00", "h", ["2024-01-01 01:00+08:00", "2024-01-01 02:00+08:00"]),
])
def test_forecast_and_exogenous_share_strictly_future_grid(origin, freq, expected):
    expected = pd.DatetimeIndex(pd.to_datetime(expected))
    actual = forecast_timestamps(pd.Series([pd.Timestamp(origin)]), 2, freq)
    pd.testing.assert_index_equal(pd.DatetimeIndex(actual), expected, check_names=False)
    frame = pd.DataFrame({"ds": expected, "x": [11., 29.]})
    aligned = align_future_exog(frame, "ds", ["x"], pd.Timestamp(origin), freq, 2)
    assert aligned["x"].tolist() == [11., 29.]


@pytest.mark.parametrize("history,horizon,freq", [
    (pd.Series([pd.Timestamp("2024-01-01")]), 2, "invalid"),
    (pd.Series([pd.NaT]), 2, "D"),
    (pd.Series([], dtype="datetime64[ns]"), 2, "D"),
    (None, 2, "D"),
    (pd.Series([pd.Timestamp("2024-01-01")]), 0, "D"),
])
def test_invalid_forecast_grid_fails(history, horizon, freq):
    with pytest.raises(ValueError):
        forecast_timestamps(history, horizon, freq)
