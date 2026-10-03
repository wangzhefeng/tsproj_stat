"""ETTm1 shell 到 Python 的调用契约；捕获参数，不启动昂贵的全量回测。"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from models.registry import MODEL_REGISTRY, create_stat_model

ROOT = Path(__file__).resolve().parents[1]
SCENE = Path("scripts/ett_small/ETTm1")


@pytest.fixture
def capture_entry(tmp_path):
    project = tmp_path / "project with spaces"
    shutil.copytree(ROOT / "scripts/ett_small", project / "scripts/ett_small")
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!/bin/bash\nexec {shlex.quote(sys.executable)} -c "
        + shlex.quote("import json, os, sys; print(json.dumps({'cwd': os.getcwd(), 'argv': sys.argv[1:]})); sys.exit(23)")
        + ' "$@"\n'
    )
    python.chmod(0o755)

    def capture(relative, *args):
        result = subprocess.run(
            ["bash", str(project / SCENE / relative), *args], cwd=tmp_path,
            env={k: v for k, v in os.environ.items() if not k.startswith("TSPROJ_")},
            capture_output=True, text=True,
        )
        assert result.returncode == 23, result.stderr
        payload = json.loads(result.stdout)
        assert payload["cwd"] == str(project.resolve())
        return payload["argv"]

    return capture


def _options(argv):
    assert argv[:2] == ["-u", "run.py"]
    return dict(zip(argv[2::2], argv[3::2], strict=True))


def test_eda_default_config_and_override(capture_entry):
    args = ["--results_dir", "output with spaces"]
    assert capture_entry("eda/run_eda.sh", *args) == [
        "-u", "run_eda.py", "--config", str(SCENE / "eda/15min.yaml"), *args,
    ]


def test_univariate_single_and_batch_share_experiment(capture_entry):
    batch = _options(capture_entry("run_models_all.sh"))
    params = json.loads(batch["--batch_models"])
    assert set(batch["--model_names"].split(",")) == set(params)
    names = {name for name, spec in MODEL_REGISTRY.items() if not spec.supports_multivariate}
    assert set(params) == names - {"neuralprophet"}
    expected = {
        "--data_path": "dataset/ETT-small/ETTm1.csv", "--time_col": "date",
        "--target_col": "OT", "--freq": "15min", "--results_data_name": "ett_small/ETTm1",
        "--history_size": "1920", "--backtest_train_size": "1920",
        "--predict_horizon": "96", "--backtest_horizon": "96", "--backtest_step": "480",
        "--backtest_window_mode": "expanding", "--forecast_strategy": "native",
        "--seasonal_period": "96", "--lags": "1,2,96,192", "--do_eda": "false",
    }
    for name in sorted(names):
        argv = capture_entry(f"univariate/run_{name}.sh", "--results_dir", "isolated output")
        assert argv[-2:] == ["--results_dir", "isolated output"]
        opts = _options(argv)
        assert opts["--model_name"] == name
        assert "--endog_cols" not in opts and "--exog_cols" not in opts
        for key, value in expected.items():
            assert opts[key] == batch[key] == value
        model_params = json.loads(opts["--model_params"])
        create_stat_model(name, model_params)
        if name in params:
            assert model_params == params[name]
        if name in {"seasonal_naive", "auto_ces", "auto_ets", "auto_theta", "dynamic_theta", "seasonal_window_average"}:
            assert model_params["season_length"] == 96
        if name == "sarima":
            assert model_params["seasonal_order"][-1] == 96
        if name == "ets":
            assert model_params["seasonal_periods"] == 96
        if name == "theta":
            assert model_params["period"] == 96
        if name == "tbats":
            assert model_params["seasonal_periods"] == [96]


@pytest.mark.parametrize("name", ["var", "bayesian_var", "linear_var", "models_all"])
def test_multivariate_inputs_survive_move(capture_entry, name):
    opts = _options(capture_entry(f"multivariate/run_{name}.sh"))
    assert opts["--endog_cols"] == "HUFL,HULL,MUFL,MULL,LUFL,LULL"
    assert opts["--data_path"] == "dataset/ETT-small/ETTm1.csv"
    assert opts["--results_data_name"] == "ett_small/ETTm1"
    assert opts["--target_col"] == "OT" and opts["--freq"] == "15min"
    assert opts["--backtest_step"] == "480"
