"""EDA 输入边界：共用质量错误、保真视图与领域样本门禁。"""
import numpy as np
import pandas as pd
import pytest

from eda.input_view import prepare_series


@pytest.fixture
def frame():
    return pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=12),
                         "y": np.arange(12, dtype=float)}, index=np.arange(20, 32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_eda_reports_nonfinite_observations(frame, value):
    frame.loc[25, "y"] = value
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="EDA input.*non-finite"):
        prepare_series(frame)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("case", ["nat", "duplicate", "reverse", "gap", "wrong_freq", "empty"])
def test_eda_rejects_invalid_time_without_repair(frame, case):
    if case == "nat":
        frame.loc[25, "ds"] = pd.NaT
    elif case == "duplicate":
        frame.loc[25, "ds"] = frame.loc[24, "ds"]
    elif case == "reverse":
        frame = frame.iloc[::-1]
    elif case == "gap":
        frame = frame.drop(index=25)
    elif case == "wrong_freq":
        frame["ds"] = pd.date_range("2026-01-01", periods=len(frame), freq="2D")
    else:
        frame = frame.iloc[:0]
    before = frame.copy(deep=True)
    with pytest.raises(ValueError):
        prepare_series(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_eda_keeps_values_time_and_domain_minimum(frame):
    before = frame.copy(deep=True)
    expected = pd.Series(np.arange(12, dtype=float),
                         index=pd.DatetimeIndex(frame.ds), name="y")
    pd.testing.assert_series_equal(prepare_series(frame), expected)
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="need >= 10 samples"):
        prepare_series(frame.iloc[:9])
