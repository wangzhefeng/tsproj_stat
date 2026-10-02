"""AIDC shell 入口与批跑契约；仅使用隔离子脚本，不启动真实模型。"""
import csv
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/aidc_power_month"


def _executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.mark.parametrize("route", ["A", "B"])
def test_eda_route_is_self_contained_and_preserves_arguments(tmp_path, route):
    project = tmp_path / "project with spaces"
    entry = project / "scripts/aidc_power_month" / f"route_{route}" / "run_eda.sh"
    entry.parent.mkdir(parents=True)
    shutil.copy2(SOURCE / f"route_{route}" / "run_eda.sh", entry)
    _executable(project / ".venv/bin/python", '#!/usr/bin/env bash\nprintf "%s\\n" "$PWD" "$LOG_NAME" "$@"\nexit 23\n')
    args = ["--config", "a configuration with spaces.yaml", "--eda_bds_mode", "off"]
    result = subprocess.run(["bash", str(entry), *args], cwd=tmp_path,
                            env={**os.environ, "ROUTE": "invalid-inherited-route"},
                            capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    assert result.stdout.splitlines() == [str(project.resolve()), f"eda_{route}_Loads", "-u", "run_eda.py",
        "--config", f"scripts/aidc_power_month/route_{route}/eda/D.yaml", *args]


@pytest.mark.parametrize("mode", ["ok", "missing_one", "missing_all", "failed_one", "missing_python"])
def test_batch_routes_summary_and_exit_status(tmp_path, mode):
    project = tmp_path / "project"
    scene = project / "scripts/aidc_power_month"
    scene.mkdir(parents=True)
    entry = scene / "run_all.sh"
    shutil.copy2(SOURCE / "run_all.sh", entry)
    if mode != "missing_python":
        _executable(project / ".venv/bin/python", "#!/usr/bin/env bash\nexit 0\n")
    # 拦截意外的依赖安装命令；环境预检不应通过反引号执行 uv sync。
    trap = project / "uv-invoked"
    _executable(project / "bin/uv", '#!/usr/bin/env bash\nprintf invoked > "$UV_TRAP"\nexit 7\n')
    for route in ("A", "B"):
        for original in (SOURCE / f"route_{route}").glob("run_*.sh"):
            if mode == "missing_all" or (mode == "missing_one" and route == "A" and original.name == "run_naive.sh"):
                continue
            fail = mode == "failed_one" and route == "A" and original.name == "run_naive.sh"
            _executable(scene / f"route_{route}" / original.name,
                        f'#!/usr/bin/env bash\nprintf "executed {route}/{original.stem}\\n"\nexit {7 if fail else 0}\n')
    result = subprocess.run(["bash", str(entry)], cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "PATH": str(project / "bin") + os.pathsep + os.environ["PATH"],
                                 "UV_TRAP": str(trap)})
    assert not trap.exists(), "environment check executed uv instead of displaying advice"
    if mode == "missing_python":
        assert result.returncode != 0
        assert not (project / "logs").exists()
        return
    summary = next((project / "logs").glob("aidc_power_month/run_all_*/summary.csv"))
    with summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 32
    assert {row["route"] for row in rows} == {"A", "B"}
    statuses = [row["status"] for row in rows]
    if mode == "ok":
        assert result.returncode == 0 and set(statuses) == {"ok"}
    else:
        assert result.returncode != 0
        if mode == "missing_all":
            assert set(statuses) == {"missing"}
        elif mode == "missing_one":
            assert statuses.count("missing") == 1 and statuses.count("ok") == 31
        else:
            assert statuses.count("FAIL(7)") == 1 and statuses.count("ok") == 31
    for row in rows:
        if row["status"] != "missing":
            assert Path(row["log"]).read_text() == f'executed {row["route"]}/{row["script"]}\n'
