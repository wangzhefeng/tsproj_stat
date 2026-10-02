"""可信本地 checkpoint 包；完成清单最后发布，加载前校验全部成员。"""
from __future__ import annotations

import datetime
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path

from artifacts.identity import file_fingerprint
from artifacts.writers import write_json, write_pickle
from models.base import BaseStatModel
from utils.log_util import logger

_KEY_DEPS = ["statsmodels", "statsforecast", "numpy", "pandas", "pmdarima", "scikit-learn"]

# 仅迁移本项目已搬动的类；不是安全沙箱，仍只允许可信本地 pickle。
_LEGACY_SF_CLASSES = {
    "models.model.baseline_models": {
        "AutoETSModel", "AutoCESModel", "AutoThetaModel", "DynamicThetaModel",
        "RandomWalkWithDriftModel", "SeasonalWindowAverageModel", "_StatsForecastModelBase",
    },
    "models.model.arima_family": {"StatsForecastAutoARIMAModel", "_StatsForecastPredict"},
}


class _ArchiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name in _LEGACY_SF_CLASSES.get(module, set()):
            module = "models.model.statsforecast_backend"
        return super().find_class(module, name)


def _collect_dep_versions(pkgs: list[str]) -> dict[str, str]:
    versions = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def _warn_if_dep_mismatch(saved_deps: dict[str, str]) -> None:
    current = _collect_dep_versions(list(saved_deps))
    for pkg, saved in saved_deps.items():
        if saved != "unknown" and current[pkg] != saved:
            logger.warning(f"[ModelLoad] dependency version mismatch: {pkg} saved={saved} current={current[pkg]}")


def _save(model, path: Path, meta: dict | None, transformer=None) -> None:
    manifest = path.parent / "checkpoint_manifest.json"
    # running 标记先写：中断的新归档不能退化成 legacy 并绕过校验。
    write_json(manifest, {"schema_version": 2, "status": "running"})
    runtime = model.runtime_info() if isinstance(model, BaseStatModel) else None
    metadata = {
        **(meta or {}), "schema_version": 2,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(), "key_deps": _collect_dep_versions(_KEY_DEPS),
        "model_class": type(model).__name__,
        "is_fallback": runtime.using_fallback_prediction if runtime else False,
        "fallback_reason": runtime.fallback_reason if runtime else None,
    }
    members = [path]
    write_pickle(path, model)
    if transformer is not None:
        transform_path = path.parent / "target_transformer.pkl"
        write_pickle(transform_path, transformer)
        metadata["target_transformer_path"] = transform_path.name
        metadata["model_output_scale"] = "transformed_target" if transformer.enabled else "original_target"
        members.append(transform_path)
    meta_path = path.parent / "model_meta.json"
    write_json(meta_path, metadata)
    members.append(meta_path)
    write_json(manifest, {"schema_version": 2, "status": "succeeded", "model": path.name,
                          "files": {p.name: file_fingerprint(p) for p in members}})


def save_model(model, path: str, meta: dict | None = None) -> None:
    _save(model, Path(path), meta)


def save_checkpoint(model, transformer, directory: Path, meta: dict) -> None:
    _save(model, directory / "model.pkl", meta, transformer)


def _verify(path: Path) -> dict:
    manifest_path = path.parent / "checkpoint_manifest.json"
    meta_path = path.parent / "model_meta.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2 or manifest.get("status") != "succeeded":
            raise ValueError("incomplete checkpoint")
        files = manifest.get("files", {})
        if manifest.get("model") != path.name or path.name not in files or "model_meta.json" not in files:
            raise ValueError("checkpoint integrity: missing members")
        for name, fingerprint in files.items():
            member = path.parent / name
            if (Path(name).name != name or not member.resolve().is_relative_to(path.parent.resolve())
                    or not member.is_file() or file_fingerprint(member) != fingerprint):
                raise ValueError(f"checkpoint integrity failure: {name}")
    else:
        logger.warning("[ModelLoad] legacy archive: no integrity guarantee; trusted local files only")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("schema_version") == 2 and not manifest_path.exists():
        raise ValueError("incomplete checkpoint: missing manifest")
    _warn_if_dep_mismatch(meta.get("key_deps", {}))
    return meta


def load_model(path: str):
    """只加载可信本地 pickle；checksum 不是来源认证。"""
    p = Path(path)
    _verify(p)
    with p.open("rb") as stream:
        return _ArchiveUnpickler(stream).load()


def load_checkpoint(directory: Path):
    path = directory / "model.pkl"
    meta = _verify(path)
    name = meta.get("target_transformer_path")
    if not name or Path(name).name != name:
        raise ValueError("checkpoint requires a relative transformer path")
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
    if name not in manifest["files"]:
        raise ValueError("checkpoint integrity: transformer not listed")
    with path.open("rb") as stream:
        model = _ArchiveUnpickler(stream).load()
    with (directory / name).open("rb") as stream:
        transformer = _ArchiveUnpickler(stream).load()
    return model, transformer, meta
