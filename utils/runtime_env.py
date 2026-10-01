"""运行时环境保障：matplotlib 可写配置目录等进程级前置条件。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def ensure_mpl_config_dir() -> str:
    """确保 matplotlib 有可写的配置/缓存目录。

    默认使用 matplotlib 自己的用户级目录（~/.matplotlib）；仅当该目录不可写时
    （受限环境，见 LOG.md P07）回退到系统临时目录。不在仓库根目录落盘任何缓存。
    """
    existing = os.environ.get("MPLCONFIGDIR")
    if existing:
        return existing

    default_dir = Path.home() / ".matplotlib"
    try:
        default_dir.mkdir(parents=True, exist_ok=True)
        probe = default_dir / ".write_probe"
        probe.touch()
        probe.unlink()
        # 默认目录可写，无需覆盖 MPLCONFIGDIR。
        return str(default_dir)
    except OSError:
        target = Path(tempfile.mkdtemp(prefix="mplconfig-"))
        os.environ["MPLCONFIGDIR"] = str(target)
        return str(target)
