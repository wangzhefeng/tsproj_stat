from __future__ import annotations

import os
from pathlib import Path


def ensure_mpl_config_dir(base_dir: str | Path | None = None) -> str:
    existing = os.environ.get("MPLCONFIGDIR")
    if existing:
        return existing

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    target = (root / ".mplconfig").resolve()
    target.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(target)
    return str(target)
