"""EDA 证据契约：失败不冒充成功、单位准确、无证据不推断。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from eda.diagnostics import run_diagnostics
from eda.report_generator import generate_eda_report


def test_seasonal_test_failure_is_not_a_success_or_zero_recommendation(monkeypatch):
    import eda.diagnostics as diagnostics
    from eda.recommendations import build_recommendations

    def fail(*args, **kwargs):
        raise RuntimeError("seasonal backend unavailable")

    monkeypatch.setattr(diagnostics, "nsdiffs", fail)
    series = pd.Series(np.random.default_rng(4).normal(size=80))
    summary, frame = run_diagnostics(series)
    rows = frame[frame.category == "seasonal_diff"]
    assert not rows.ok.any()
    assert rows.error.str.contains("seasonal backend unavailable").all()
    assert build_recommendations(summary, frame, 7)["differencing"]["recommended_D"] is None


def test_report_uses_sampling_units_and_does_not_invent_direction(tmp_path):
    summary = {
        "n_samples": 1000,
        "decomposition": {"period": 96, "trend_strength": 0.99, "seasonal_strength": 0.1},
        "cycle": {"acf_peak_lags": [87, 182]},
        "seasonal_diff": {"D_ch": None, "D_ocsb": None},
        "stationarity": [{"name": "adf", "ok": False, "error": "test failure"}],
    }
    (tmp_path / "eda_summary.json").write_text(json.dumps(summary))
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots/seasonal_subseries.png").write_bytes(b"present")
    report = generate_eda_report(tmp_path, cfg=AppConfig(freq="15min"))
    assert report is not None
    text = Path(report).read_text()
    assert "21.75 小时" in text and "45.5 小时" in text
    assert "天滞后" not in text and "显著上升" not in text
    assert "稳定且可用" not in text and "一致指向" not in text
    assert "证据不足" in text


def test_short_decomposition_is_not_success():
    _, frame = run_diagnostics(pd.Series(np.arange(12.0)), period=7)
    assert not frame.loc[frame.category == "decomposition", "ok"].any()


def test_pure_linear_trend_does_not_have_perfect_seasonal_strength():
    from eda.diagnostics import decomposition_report
    report, _ = decomposition_report(pd.Series(100 + np.arange(240) * 0.1), period=24)
    assert report["trend_strength"] > 0.99
    assert report["seasonal_strength"] < 0.01


def test_shared_computation_and_numeric_exports(tmp_path, monkeypatch):
    import eda.diagnostics as diagnostics
    from eda.pipeline import run_eda
    from statsmodels.tsa.seasonal import STL

    calls = []
    kernels = {name: [] for name in ("acf", "pacf", "periodogram")}
    for name, counts in kernels.items():
        original = getattr(diagnostics, name)
        def observe(*args, _fn=original, _counts=counts, **kwargs):
            _counts.append(1)
            return _fn(*args, **kwargs)
        monkeypatch.setattr(diagnostics, name, observe)
    def counted(*args, **kwargs):
        calls.append(1)
        return STL(*args, **kwargs)
    monkeypatch.setattr(diagnostics, "STL", counted)
    t = np.arange(240)
    values = 100 + t * 0.2 + 5 * np.sin(2 * np.pi * t / 24)
    values[121] += 50
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=len(t), freq="h"), "value": values})
    out = run_eda(frame, "time", "value", "h", str(tmp_path), period=24, nlags=48, bds_mode="off")
    assert len(calls) == 1
    assert all(len(counts) == 3 for counts in kernels.values())  # 每视图一次，绘图不重算。
    components = pd.read_csv(out["eda_components_path"])
    np.testing.assert_allclose(components[["trend", "seasonal", "residual"]].sum(axis=1), values)
    summary = json.loads(Path(out["eda_summary_path"]).read_text())
    assert {"raw", "linear_detrended", "difference"} == set(summary["views"])
    correlations = pd.read_csv(out["eda_correlations_path"])
    raw = correlations[correlations.view == "raw"]
    assert raw.lag.max() == 48
    assert raw.acf.iloc[1] == pytest.approx(np.corrcoef(values[:-1], values[1:])[0, 1], abs=0.03)
    anomalies = pd.read_csv(out["eda_outliers_path"])
    spike = anomalies[(anomalies.method == "stl_residual_mad") & (anomalies.time == str(frame.time.iloc[121]))]
    assert spike.value.iloc[0] == values[121]
    assert summary["stochasticity"]["status"] == "skipped"
    report = generate_eda_report(tmp_path, cfg=AppConfig(freq="h"))
    assert report is not None
    text = Path(report).read_text()
    assert "raw/bds_dim2" in text and "skipped" in text and "explicitly disabled" in text


def test_period_validation_distinguishes_trend_and_repeatable_signal():
    from eda.evidence import validate_periods, period_candidates

    t = np.arange(24 * 56)
    trend = pd.Series(100 + 0.1 * t)
    signal = trend + 5 * np.sin(2 * np.pi * t / 24)
    assert not any(r["stable"] for r in validate_periods(trend, [24, 168]))
    daily = validate_periods(signal, [24])[0]
    assert daily["stable"] and daily["forward_r2"] > 0.95
    assert period_candidates(24, "h", []) == [24, 168]
    assert period_candidates(7, "D", []) == [7]
    assert period_candidates(12, "MS", []) == [12]
    changed = signal.copy()
    changed.iloc[len(t)//2:] = trend.iloc[len(t)//2:] + 5 * np.cos(2 * np.pi * t[len(t)//2:] / 24)
    assert not validate_periods(changed, [24])[0]["stable"]
    assert validate_periods(signal.iloc[:60], [24])[0]["status"] == "insufficient"


def test_bds_explicit_range_and_resource_limit(monkeypatch):
    import eda.diagnostics as diagnostics

    series = pd.Series(np.random.default_rng(1).normal(size=80),
                       index=pd.date_range("2024-01-01", periods=80, freq="h"))
    original = diagnostics.bds
    seen = []
    def observed(values, **kwargs):
        seen.append(values.copy())
        return original(values, **kwargs)
    monkeypatch.setattr(diagnostics, "bds", observed)
    limited = diagnostics.stochasticity_report(series, mode="full", max_samples=40)
    assert limited["status"] == "resource_limit" and not seen
    tail = diagnostics.stochasticity_report(series, mode="tail", max_samples=40)
    np.testing.assert_array_equal(seen[0], series.iloc[-40:].to_numpy())
    assert tail["n_samples"] == 40 and tail["start"] == str(series.index[-40])
    full = diagnostics.stochasticity_report(series)
    np.testing.assert_array_equal(seen[1], series.to_numpy())
    assert full["n_samples"] == 80


def test_no_period_recommended_from_trend_only(tmp_path):
    from eda.pipeline import run_eda
    n = 240
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=n, freq="h"),
                          "value": 100 + np.arange(n) * 0.1})
    out = run_eda(frame, "time", "value", "h", str(tmp_path), period=24, save_plots=False, bds_mode="off")
    rec = json.loads(Path(out["eda_recommendations_path"]).read_text())
    assert rec["seasonal_period"]["recommended_period"] is None


@pytest.mark.parametrize("mode,limit", [("invalid", 0), ("full", -1), ("tail", 0), ("tail", 9)])
def test_bds_config_rejects_invalid_policy(mode, limit):
    with pytest.raises(ValueError, match="BDS"):
        AppConfig(eda_bds_mode=mode, eda_bds_max_samples=limit).validate()


def test_constant_input_keeps_reportable_failures_not_fake_success(tmp_path):
    from eda.pipeline import run_eda
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=40, freq="D"), "value": 10.0})
    out = run_eda(frame, "time", "value", "D", str(tmp_path), bds_mode="off")
    summary = json.loads(Path(out["eda_summary_path"]).read_text())
    assert summary["decomposition"]["seasonal_strength"] == 0
    rows = pd.read_csv(out["eda_diagnostics_path"])
    assert not rows.loc[rows.category == "cycle", "ok"].any()
    assert not rows.loc[rows.category == "forecastability", "ok"].any()
    rec = json.loads(Path(out["eda_recommendations_path"]).read_text())
    assert rec["differencing"]["recommended_d"] is None
