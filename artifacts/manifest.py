"""一次运行的完成协议；只枚举显式登记的本次目录，排除累计监控。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from config import AppConfig
from artifacts.identity import build_identity, file_fingerprint
from artifacts.writers import write_json


def verify_manifest(path: Path, root: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["files"]:
        target = root / entry["path"]
        if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError(f"manifest integrity: invalid member {target}")
        if file_fingerprint(target) != {key: entry[key] for key in ("sha256", "size_bytes")}:
            raise ValueError(f"manifest integrity failure: {target}")
    return payload


class RunManifest:
    def __init__(self, path: Path, cfg: AppConfig, run_id: str, *, source: dict | None = None):
        self.path = path
        self.root = Path(cfg.results_dir).resolve()
        self.payload = {
            "schema_version": 2, "run_id": run_id, "status": "running", "config": asdict(cfg),
            "identity": asdict(build_identity(cfg, eda=cfg.is_eda_only(), source=source)),
            "inputs": {}, "files": [], "errors": {},
            "stages": {stage: "pending" if getattr(cfg, f"do_{stage}") else "skipped"
                       for stage in ("eda", "train", "test", "forecast")},
        }
        write_json(path, self.payload)

    def inputs(self, inputs: dict) -> None:
        self.payload["inputs"] = inputs
        write_json(self.path, self.payload)

    def finish(self, outputs: dict, directories: list[Path], *, fatal: str | None = None) -> str:
        errors = {key: value for key, value in outputs.items() if key.split("::", 1)[0].endswith("_error")}
        if outputs.get("multi_model") != "true" and fatal is None:
            required = [("train", "model_path"), ("test", "test_metrics_path"), ("forecast", "prediction_path")]
            config = self.payload["config"]
            if config["simulate_enabled"]:
                required.append(("forecast", "simulated_paths_path"))
                if config["simulate_quantiles"]:
                    required.append(("forecast", "simulated_quantiles_path"))
            for stage, key in required:
                target = outputs.get(key)
                if self.payload["stages"][stage] != "skipped":
                    if not isinstance(target, str) or not Path(target).is_file():
                        errors.setdefault(f"{stage}_error", f"missing required output: {key}")
        if fatal is not None:
            errors["run_error"] = fatal
        self.payload["errors"] = errors
        for stage, status in self.payload["stages"].items():
            if status != "skipped":
                self.payload["stages"][stage] = (
                    "failed" if any(stage in key for key in errors) else
                    "not_completed" if fatal is not None else "succeeded"
                )
        paths = {p.resolve() for directory in directories if directory.exists()
                 for p in directory.rglob("*") if p.is_file() and p.name != "run_manifest.json"}
        files = []
        for p in sorted(paths):
            if not p.is_relative_to(self.root):
                raise ValueError(f"artifact outside results root: {p}")
            if p.name.startswith(".artifact-"):
                raise ValueError(f"unfinished temporary artifact: {p}")
            files.append({"path": str(p.relative_to(self.root)), **file_fingerprint(p)})
        self.payload["files"] = files
        self.payload["outputs"] = outputs
        self.payload["status"] = "failed" if errors else "succeeded"
        # 先在内存核验；完成标记最后发布，不短暂暴露未经核验的 succeeded。
        for entry in files:
            if file_fingerprint(self.root / entry["path"]) != {k: entry[k] for k in ("sha256", "size_bytes")}:
                raise ValueError("artifact changed during manifest publication")
        write_json(self.path, self.payload)
        return str(self.path)
