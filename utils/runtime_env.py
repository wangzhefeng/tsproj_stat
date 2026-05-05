from __future__ import annotations

import os
from pathlib import Path


def ensure_mpl_config_dir(base_dir: str | Path | None = None) -> str:
    """确保 matplotlib 使用项目内缓存目录。

    受限环境下默认用户目录可能不可写；在入口启动早期设置 MPLCONFIGDIR，
    可以避免 EDA/可视化阶段产生无关缓存告警。
    """
    existing = os.environ.get("MPLCONFIGDIR")
    if existing:
        return existing

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    target = (root / ".mplconfig").resolve()
    target.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(target)
    return str(target)
