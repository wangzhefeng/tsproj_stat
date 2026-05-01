import warnings
from pathlib import Path

from app.runtime import configure_warnings, ensure_mpl_config_dir


def test_configure_warnings_smoke():
    with warnings.catch_warnings():
        configure_warnings()


def test_ensure_mpl_config_dir_sets_project_local_path(monkeypatch, tmp_path):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.chdir(tmp_path)

    configured = ensure_mpl_config_dir()

    assert configured == str((tmp_path / ".mplconfig").resolve())
    assert Path(configured).exists()
