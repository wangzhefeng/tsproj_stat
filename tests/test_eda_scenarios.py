"""数据场景契约：任务未确认不产配置，审计只读不重建。"""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from eda.config_advice import model_config_section


def test_unconfirmed_task_never_turns_defaults_into_yaml():
    text = "\n".join(model_config_section({"n_samples": 1000}, {"differencing": {"recommended_d": 1}}, AppConfig(data_path="input.csv")))
    assert "```yaml" not in text
    assert "预测任务未确认" in text


def test_confirmed_task_candidates_share_window_and_identity():
    import yaml
    import re
    from artifacts.paths import build_eda_path
    cfg = AppConfig(data_path="input.csv", freq="h", history_size=100, predict_horizon=12, eda_task_confirmed=True)
    summary = {"n_samples": 1000, "decomposition": {"period": 24, "trend_strength": 0.9}}
    rec = {"differencing": {"recommended_d": 1, "recommended_D": 0}, "seasonal_period": {"recommended_period": 24}}
    configs = [yaml.safe_load(s) for s in re.findall(r"```yaml\n(.*?)```", "\n".join(model_config_section(summary, rec, cfg)), re.S)]
    assert len(configs) >= 3
    assert all(c["history_size"] == c["backtest_train_size"] == 100 for c in configs)
    assert build_eda_path(cfg) != build_eda_path(replace(cfg, predict_horizon=24))
    short = replace(cfg, history_size=20)
    text = "\n".join(model_config_section(summary, rec, short))
    assert "model_name: seasonal_naive" not in text


def test_audit_verification_readonly_and_tamper_detection(tmp_path):
    from data_provider.resampling.service import aggregate_csv
    from data_provider.resampling.provenance import inspect_aggregation_audit
    source = tmp_path / "source.csv"
    pd.DataFrame({"t": pd.date_range("2024-01-01", periods=96, freq="15min"), "y": np.arange(96.0)}).to_csv(source, index=False)
    result = aggregate_csv(source_path=source, time_col="t", target_col="y", source_freq="15min", target_freq="h", fill_method="none", output_path=tmp_path/"derived.csv")
    before = {p: p.read_bytes() for p in (source, result.data_path, result.audit_path)}
    info = inspect_aggregation_audit(result.data_path, freq="h", time_col="t", target_col="y")
    assert info["status"] == "verified" and info["fill_uses_future"] is False
    assert all(p.read_bytes() == b for p,b in before.items())
    result.data_path.write_bytes(before[result.data_path] + b"\n")
    assert inspect_aggregation_audit(result.data_path, freq="h", time_col="t", target_col="y")["status"] == "invalid"
    plain = tmp_path / "plain.csv"
    plain.write_bytes(before[result.data_path])
    assert inspect_aggregation_audit(plain, freq="h", time_col="t", target_col="y")["status"] == "missing"
    assert not plain.with_name(plain.name + ".aggregate.json").exists()


def test_segment_statistics_and_local_event_boundaries():
    from eda.windows import summarize_windows, local_outliers, summarize_events
    times = pd.date_range("2024-01-01", periods=240, freq="h")
    trend = pd.Series(100 + np.arange(240) * 0.5, index=times)
    monthly, rolling, periods = summarize_windows(trend, [24], 96, 48)
    assert monthly.iloc[0]["mean"] == pytest.approx(trend.mean())
    assert np.allclose(rolling.slope_per_point, 0.5)
    assert rolling.iloc[-1]["end"] == str(times[-1])
    assert (rolling.n_samples == 96).all()
    assert not periods.stable.any()
    residual = pd.Series(np.tile([-1., 1.], 120), index=times)
    residual.iloc[180:] *= 10
    residual.iloc[210:212] = 100
    flags = local_outliers(trend, residual, 21)
    assert str(times[200]) not in set(flags.time)
    assert {str(times[210]), str(times[211])} <= set(flags.time)
    events = summarize_events(flags, trend)
    event = events[events.start == str(times[210])].iloc[0]
    assert event["n_points"] == 2 and event["end"] == str(times[211])
    assert event["duration_seconds"] == 7200
    assert event["peak_value"] in trend.iloc[210:212].tolist()


def test_eda_pipeline_exports_scenario_evidence(tmp_path):
    from eda.pipeline import run_eda
    frame = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=240, freq="h"),
                          "y": 100 + np.arange(240) * 0.1 + np.sin(np.arange(240)*2*np.pi/24)})
    path = tmp_path/"input.csv"
    frame.to_csv(path, index=False)
    out = run_eda(frame, "t", "y", "h", str(tmp_path/"out"), period=24, nlags=12,
                  acf_nlags=168, window_size=96, window_step=48, local_outlier_window=21,
                  source_path=str(path), bds_mode="off", save_plots=False)
    corr = pd.read_csv(out["eda_correlations_path"])
    raw = corr[corr.view == "raw"]
    assert raw.lag.max() == 168 and raw.loc[raw.lag>12, "pacf"].isna().all()
    monthly = pd.read_csv(out["eda_monthly_path"])
    assert monthly.iloc[0]["mean"] == pytest.approx(frame.y.mean())
    assert pd.read_csv(out["eda_rolling_path"]).n_samples.eq(96).all()
    s = json.loads(Path(out["eda_summary_path"]).read_text())
    assert s["source_provenance"]["status"] == "missing"
    assert not path.with_name(path.name+".aggregate.json").exists()


def test_scenario_configs_and_script_override(tmp_path):
    import subprocess
    import os
    from config.loader import load_config
    root = Path(__file__).resolve().parents[1]
    scenarios = sorted((root/"scripts").glob("**/eda/*.yaml"))
    assert len(scenarios) == 7
    for path in scenarios:
        cfg = load_config(path)
        assert cfg.is_eda_only() and not cfg.aggregation_enabled
        assert cfg.eda_bds_mode == "off" and not cfg.eda_task_confirmed
    cfg = load_config(root/"scripts/aidc_power_month/route_A/eda/15min.yaml")
    assert cfg.freq == "15min" and cfg.eda_period == 96 and cfg.eda_acf_nlags == 672
    frame = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=60, freq="h"), "y": np.arange(60.) + np.sin(np.arange(60.))})
    source = tmp_path/"override.csv"
    frame.to_csv(source, index=False)
    env = {k:v for k,v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("TSPROJ_")}
    completed = subprocess.run(["bash", str(root/"scripts/aidc_power_month/route_A/run_eda.sh"),
        "--config", str(root/"scripts/aidc_power_month/route_A/eda/15min.yaml"), "--data_path", str(source),
        "--time_col", "t", "--target_col", "y", "--freq", "h", "--eda_period", "12",
        "--eda_nlags", "12", "--eda_acf_nlags", "24", "--eda_window_size", "24", "--eda_window_step", "12",
        "--eda_local_outlier_window", "7", "--results_dir", str(tmp_path/"results")],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    summaries = list((tmp_path/"results").rglob("eda_summary.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text())
    assert payload["n_samples"] == 60 and payload["freq"] == "h"
