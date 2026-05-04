from __future__ import annotations

import datetime
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path

from utils.log_util import logger


def _collect_dep_versions(pkgs: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


_KEY_DEPS = ["statsmodels", "numpy", "pandas", "pmdarima", "scikit-learn"]


def _warn_if_dep_mismatch(saved_deps: dict[str, str]) -> None:
    current = _collect_dep_versions(list(saved_deps.keys()))
    for pkg, saved_ver in saved_deps.items():
        cur_ver = current.get(pkg, "unknown")
        if cur_ver != saved_ver and saved_ver != "unknown":
            logger.warning(
                f"[ModelLoad] dependency version mismatch: {pkg} "
                f"saved={saved_ver} current={cur_ver}"
            )


def save_model(model, path: str, meta: dict | None = None) -> None:
    """保存模型到 path，同时将元数据写入同目录的 model_meta.json。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f)

    full_meta: dict = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "key_deps": _collect_dep_versions(_KEY_DEPS),
        "is_fallback": getattr(model, "_is_fallback", False),
        "fallback_reason": getattr(model, "_fallback_reason", None),
        "model_class": type(model).__name__,
    }
    if meta:
        full_meta.update(meta)

    meta_path = p.parent / "model_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2, default=str, ensure_ascii=False)


def load_model(path: str):
    """加载模型，并校验依赖版本兼容性（仅 warning，不阻断）。"""
    p = Path(path)
    meta_path = p.parent / "model_meta.json"
    if meta_path.exists():
        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            _warn_if_dep_mismatch(meta.get("key_deps", {}))
            logger.info(
                f"[ModelLoad] loading {meta.get('model_class','?')} "
                f"saved at {meta.get('saved_at','?')}"
            )
        except Exception as exc:
            logger.warning(f"[ModelLoad] failed to read model_meta.json: {exc}")
    with p.open("rb") as f:
        return pickle.load(f)
