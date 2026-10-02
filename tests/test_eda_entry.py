"""独立 EDA 入口：配置解析、模型阶段隔离、非仓库 cwd 和失败退出。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def scenario(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame({"t": pd.date_range("2024-01-01", periods=60, freq="h"),
                  "y": 10 + np.arange(60)*0.1 + np.sin(np.arange(60))}).to_csv(source, index=False)
    payload = {"data_path": str(source), "time_col": "t", "target_col": "y", "freq": "h",
               "eda_period": 12, "eda_nlags": 12, "eda_bds_mode": "off",
               "results_dir": str(tmp_path/"results")}
    config = tmp_path / "eda.yaml"
    config.write_text(yaml.safe_dump(payload))
    return config, payload


def test_generic_runner_reads_config_without_model_stages(tmp_path, monkeypatch):
    from eda.runner import run_from_config
    from pipeline.runner import ModelApp
    from artifacts.manifest import verify_manifest
    config, payload = scenario(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("model stage must not run")
    for name in ("train", "test", "forecast", "_run_auto_select"):
        monkeypatch.setattr(ModelApp, name, forbidden)
    result = run_from_config(config)
    manifest = verify_manifest(Path(result["manifest_path"]), Path(payload["results_dir"]))
    assert manifest["status"] == "succeeded"
    assert manifest["stages"] == {"eda": "succeeded", "train": "skipped", "test": "skipped", "forecast": "skipped"}
    summary = json.loads(Path(result["eda_summary_path"]).read_text())
    expected = pd.read_csv(payload["data_path"]).y.mean()
    assert summary["mean"] == pytest.approx(expected)
    assert not list(Path(payload["results_dir"]).rglob("model.pkl"))


def test_root_entry_runs_from_other_cwd_and_reports_failure(tmp_path):
    config, payload = scenario(tmp_path)
    env = {k:v for k,v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("TSPROJ_")}
    command = [sys.executable, str(ROOT/"run_eda.py"), "--config", str(config)]
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = next(Path(payload["results_dir"]).rglob("EDA_REPORT.md"))
    assert "预测任务未确认" in report.read_text()
    missing = subprocess.run(command+["--data_path", str(tmp_path/"missing.csv")], cwd=tmp_path,
                             env=env, capture_output=True, text=True)
    assert missing.returncode != 0
    manifests = [json.loads(p.read_text()) for p in Path(payload["results_dir"]).rglob("run_manifest.json")]
    assert {m["status"] for m in manifests} == {"succeeded", "failed"}


@pytest.mark.parametrize("overrides", [{"do_train":True},{"do_test":True},{"do_forecast":True},
    {"aggregation_enabled":True,"aggregation_source_freq":"15min"},{"auto_select":True},
    {"monitor_enabled":True},{"simulate_enabled":True},{"model_names":["naive","arima"]}])
def test_eda_entry_rejects_non_eda_tasks_before_writes(tmp_path, overrides):
    from eda.runner import run_from_config
    config, payload = scenario(tmp_path)
    with pytest.raises(ValueError, match="EDA-only"):
        run_from_config(config, overrides)
    assert not Path(payload["results_dir"]).exists()
