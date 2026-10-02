"""运行产物路径构建：RunArtifacts 与 experiment_path/eda_path。"""
from __future__ import annotations

import re
import hashlib
import json
import uuid
import fcntl
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from config import AppConfig
from config.model_params import resolve_model_params
from artifacts.identity import build_identity, effective_config, canonical_json
from artifacts.writers import write_json


# ##############################
# 准备运行结果参数
# ##############################
@dataclass(frozen=True)
class RunArtifacts:
    """一次运行对应的标准产物目录集合。"""
    setting: str
    experiment_path: Path
    eda_path: Path
    data_name: str
    checkpoints_dir: Path
    train_results_dir: Path
    test_results_dir: Path
    forecast_results_dir: Path
    eda_dir: Path
    monitor_dir: Path
    custom_monitor_dir: Path
    run_id: str


def ensure_output_dirs(cfg: AppConfig) -> None:
    """创建统一结果根；data_name 与实验子目录由 prepare_run_artifacts 负责。"""
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)


def resolve_data_name(cfg) -> str:
    """从数据路径提取数据名；显式 results_data_name 优先（支持层级路径）；demo 数据使用固定名称便于结果归类。"""
    explicit = getattr(cfg, "results_data_name", None)
    if explicit is not None:
        raw = explicit.strip()
        # 显式名只允许相对层级路径：绝对路径与上跳段直接拒绝，不做静默清洗
        if (not raw or raw.startswith(("/", "~")) or "\\" in raw
                or any(part in ("", ".", "..") for part in raw.split("/"))
                or any(ord(char) < 32 or ord(char) == 127 for char in raw)):
            raise ValueError(f"invalid results_data_name: {explicit!r}")
        return raw
    if cfg.data_path is None:
        return "demo_series"
    return path_token(Path(cfg.data_path).stem)


def _mapping_tokens(mapping: dict, prefix: str = "") -> list[str]:
    tokens: list[str] = []
    for key in sorted(mapping):
        full_key = f"{prefix}.{key}" if prefix else str(key)
        value = mapping[key]
        if isinstance(value, dict):
            tokens.extend(_mapping_tokens(value, full_key))
        else:
            tokens.append(f"{path_token(full_key)}_{path_token(value)}")
    return tokens


def path_token(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, Path):
        value = value.name
    if isinstance(value, (list, tuple)):
        return "+".join(path_token(item) for item in value) or "none"
    if isinstance(value, dict):
        return "+".join(_mapping_tokens(value)) or "default"
    text = str(value)
    if "/" in text or "\\" in text:
        text = Path(text).name
    text = re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-")
    return text or "none"


def _bounded(segment: str) -> str:
    if len(segment.encode()) <= 180:
        return segment
    return segment.encode()[:150].decode(errors="ignore") + "-" + hashlib.sha256(segment.encode()).hexdigest()[:20]


def build_experiment_path(cfg: AppConfig, *, source: dict | None = None) -> Path:
    """用完整可读参数构建稳定的模型实验相对路径。"""
    identity = build_identity(cfg, source=source)
    cfg = effective_config(cfg)
    resolved_params = resolve_model_params(cfg)
    params = path_token(resolved_params) if resolved_params else "default"
    scale = cfg.scaler_type if cfg.scale else "off"
    process = (
        f"process-denoise-{path_token(cfg.denoise_method)}_w-{cfg.denoise_window}"
        f"_detrend-{path_token(cfg.detrend_method)}"
        f"_decomp-{path_token(cfg.decomposition_method)}"
        f"_period-{path_token(cfg.seasonal_period)}"
        f"{('_periods-' + path_token(cfg.seasonal_periods)) if cfg.seasonal_periods else ''}"
    )
    readable = Path(
        f"{path_token(cfg.model_name)}-{path_token(cfg.setting_strategy_label())}",
        f"params-{params}",
        f"hist-{cfg.history_size}_pred-{cfg.predict_horizon}",
        (
            f"bt-{path_token(cfg.resolved_backtest_window_mode())}"
            f"_train-{cfg.resolved_backtest_train_size()}"
            f"_h-{cfg.backtest_horizon}_step-{cfg.backtest_step}"
            + (f"_refit-{cfg.backtest_refit_every}" if cfg.backtest_refit_every != 1 else "")
        ),
        (
            f"input-feature-{path_token(cfg.feature_mode)}"
            f"_lags-{path_token(cfg.lags)}_scale-{path_token(scale)}"
            + ("_ignore-unsupported-on" if cfg.ignore_unsupported_inputs else "")
        ),
        process,
        f"interval-{'on' if cfg.return_intervals else 'off'}_levels-{path_token(cfg.interval_levels)}"
        + (f"_conformal-{cfg.conformal_n_windows}" if cfg.return_intervals and cfg.interval_method == "conformal" else ""),
    )
    return Path(*(_bounded(part) for part in readable.parts), identity.token)


def build_eda_path(cfg: AppConfig, *, source: dict | None = None) -> Path:
    """构建只依赖数据准备与 EDA 参数的相对路径。"""
    identity = build_identity(cfg, eda=True, source=source)
    return Path(_bounded(f"{path_token(cfg.freq)}_{identity.token}"))


def plan_run_artifacts(cfg: AppConfig, run_id: str, *, source: dict | None = None) -> RunArtifacts:
    """按 data_name 和完整参数路径创建本次运行的产物目录。"""
    # 提取数据名称；层级 data_name 支持把数据项目组织成子树（非法值在 resolve_data_name 内拒绝）
    data_name = resolve_data_name(cfg)
    setting = f"{cfg.model_name}-{cfg.setting_strategy_label()}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid run_id")
    experiment_path = build_experiment_path(cfg, source=source)
    eda_path = build_eda_path(cfg, source=source)
    data_root = Path(cfg.results_dir) / data_name
    run = Path("runs") / run_id
    artifacts = RunArtifacts(
        setting=setting,
        experiment_path=experiment_path,
        eda_path=eda_path,
        data_name=data_name,
        checkpoints_dir=data_root / "checkpoints" / experiment_path / run,
        train_results_dir=data_root / "results_train" / experiment_path / run,
        test_results_dir=data_root / "results_test" / experiment_path / run,
        forecast_results_dir=data_root / "results_forecast" / experiment_path / run,
        eda_dir=data_root / "results_eda" / eda_path / run,
        monitor_dir=data_root / "monitor" / experiment_path,
        custom_monitor_dir=data_root / "custom_monitor" / experiment_path,
        run_id=run_id,
    )
    root = Path(cfg.results_dir).resolve()
    for key, value in vars(artifacts).items():
        if isinstance(value, Path) and key not in ("experiment_path", "eda_path"):
            if not value.resolve().is_relative_to(root):
                raise ValueError("artifact path escapes results_dir")
    return artifacts


def prepare_run_artifacts(cfg: AppConfig, run_id: str | None = None, *, source: dict | None = None) -> RunArtifacts:
    """顶层传入 run_id；独立调用时生成一次，不复用旧运行。"""
    artifacts = plan_run_artifacts(cfg, run_id or uuid.uuid4().hex, source=source)
    # 创建结果目录：EDA-only 仅创建 EDA 目录，避免散落空模型目录与无意义实验路径。
    dirs = (
        [artifacts.eda_dir]
        if cfg.is_eda_only()
        else [artifacts.forecast_results_dir]
        + ([artifacts.checkpoints_dir, artifacts.train_results_dir] if cfg.do_train else [])
        + ([artifacts.test_results_dir] if cfg.do_test else [])
        + ([artifacts.eda_dir] if cfg.do_eda else [])
        + ([artifacts.monitor_dir, artifacts.custom_monitor_dir] if cfg.monitor_enabled else [])
    )
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
        identity = build_identity(cfg, eda=path == artifacts.eda_dir, source=source)
        experiment_root = path.parent.parent if path.name == artifacts.run_id else path
        identity_path = experiment_root / "identity.json"
        payload = json.loads(canonical_json({"digest": identity.digest, "identity": identity.payload}))
        # 锁文件常驻；不能 unlink，否则不同进程可能锁到不同 inode。
        with (experiment_root / ".identity.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if identity_path.exists():
                    if json.loads(identity_path.read_text(encoding="utf-8")) != payload:
                        raise ValueError(f"artifact identity collision: {identity_path}")
                else:
                    write_json(identity_path, payload)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    
    return artifacts
