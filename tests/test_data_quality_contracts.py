"""质量报告不修数据；时间契约在建模前阻断，数值缺失仍按窗口修复。"""
import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from data_provider.loading.loader import DataLoader
from pipeline.runner import ModelApp
from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory
from data_provider.quality.checks import require_finite, require_regular_time


@pytest.mark.parametrize("times", [[], [pd.NaT], ["2026-01-02", "2026-01-01"]])
def test_time_primitive_rejects_empty_missing_and_reverse(times):
    with pytest.raises(ValueError, match="timestamps"):
        require_regular_time(pd.DatetimeIndex(times), "D")


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-np.inf]])
def test_finite_primitive_rejects_empty_and_invalid(values):
    with pytest.raises(ValueError, match="non-finite"):
        require_finite(pd.Series(values, dtype=float), "test")


def test_time_primitive_accepts_calendar_frequency_without_mutation():
    times = pd.date_range("2026-01-31", periods=4, freq="ME")
    require_regular_time(times, "ME")
    require_regular_time(times)
    require_finite(pd.Series([0., -2., 4.]), "test")


@pytest.mark.parametrize("dates", [
    ["2026-01-01", "2026-01-01", "2026-01-03"],
    ["2026-01-01", "2026-01-02", "2026-01-04"],
])
def test_model_preparation_rejects_bad_time_before_transform(tmp_path, dates):
    raw = pd.DataFrame({"ds": pd.to_datetime(dates), "y": [1., 2., 3.]})
    cfg = AppConfig(model_name="naive", history_size=3, predict_horizon=1,
                    results_dir=str(tmp_path), validate_freq=False)
    app = ModelApp(cfg, data_frame=raw)
    with pytest.raises(ValueError, match="timestamps"):
        app._prepare_target_series(app.loader.load_data())


def test_quality_reports_each_selected_column_without_repair():
    raw = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=4),
                        "y": [1., None, 3., 4.], "temp": [None, "bad", np.inf, "4"]})
    loader = DataLoader(None, data_frame=raw, value_cols=["y", "temp"])
    actual = loader.load_data()
    assert actual["temp"].isna().tolist() == [True, True, True, False]
    assert loader.quality_report is not None
    report = loader.quality_report.to_dict()
    assert report["missing_by_column"] == {"y": 1, "temp": 3}
    assert report["raw_missing_by_column"] == {"y": 1, "temp": 1}
    assert report["coercion_failed_by_column"] == {"y": 0, "temp": 1}
    assert report["nonfinite_by_column"] == {"y": 0, "temp": 1}


def test_direct_backtest_rejects_irregular_time_before_model_creation():
    df = pd.DataFrame({"ds": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"]),
                       "y": [1., 2., 3., 4.]})
    calls = []
    def builder():
        calls.append(True)
        return ModelFactory().create_model("naive")
    with pytest.raises(ValueError, match="timestamps"):
        rolling_backtest(df, builder, time_col="ds", train_size=2, horizon=1)
    assert not calls
