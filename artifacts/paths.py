"""运行产物路径构建：RunArtifacts 与 experiment_path/eda_path。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from config import AppConfig


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


def resolve_data_name(cfg) -> str:
    """从数据路径提取数据名；显式 results_data_name 优先（支持层级路径）；demo 数据使用固定名称便于结果归类。"""
    explicit = getattr(cfg, "results_data_name", None)
    if explicit:
        raw = explicit.strip()
        # 显式名只允许相对层级路径：绝对路径与上跳段直接拒绝，不做静默清洗
        if raw.startswith(("/", "~")) or ".." in Path(raw).parts:
            raise ValueError(f"invalid results_data_name: {explicit!r}")
        return raw.strip("/")
    if cfg.data_path is None:
        return "demo_series"
    return Path(cfg.data_path).stem


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


def resolve_model_params(cfg: AppConfig) -> dict[str, Any]:
    """返回实际交给模型工厂的参数，供实验路径和训练共同使用。"""
    params = dict(cfg.model_params)
    if cfg.model_name == "ets":
        params.setdefault("tune_smoothing_params", cfg.ets_tune_smoothing_params)
        params.setdefault("smoothing_grid_level", cfg.ets_smoothing_grid_level)
        params.setdefault("smoothing_grid_trend", cfg.ets_smoothing_grid_trend)
        params.setdefault("smoothing_grid_seasonal", cfg.ets_smoothing_grid_seasonal)
        params.setdefault("validation_size", cfg.ets_validation_size)
        if cfg.seasonal_period is not None:
            params.setdefault("seasonal_periods", cfg.seasonal_period)
    return params


def build_experiment_path(cfg: AppConfig) -> Path:
    """用完整可读参数构建稳定的模型实验相对路径。"""
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
    return Path(
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
        f"interval-{'on' if cfg.return_intervals else 'off'}_alpha-{path_token(cfg.interval_alpha)}"
        + (f"_conformal-{cfg.conformal_n_windows}" if cfg.return_intervals and cfg.interval_method == "conformal" else ""),
    )


def build_eda_path(cfg: AppConfig) -> Path:
    """构建只依赖数据准备与 EDA 参数的相对路径。"""
    aggregation = cfg.aggregation_method if cfg.aggregation_enabled else "none"
    fill_method = cfg.aggregation_fill_method if cfg.aggregation_enabled else "none"
    return Path(
        f"freq-{path_token(cfg.freq)}",
        f"period-{cfg.eda_period}_nlags-{cfg.eda_nlags}",
        (
            f"recommend-{'on' if cfg.eda_recommendation_enabled else 'off'}"
            f"_preprocessed-{'on' if cfg.eda_run_preprocessed else 'off'}"
        ),
        f"aggregation-{path_token(aggregation)}_fill-{path_token(fill_method)}",
    )


def prepare_run_artifacts(cfg: AppConfig) -> RunArtifacts:
    """按 data_name 和完整参数路径创建本次运行的产物目录。"""
    # 提取数据名称；层级 data_name 支持把数据项目组织成子树（非法值在 resolve_data_name 内拒绝）
    data_name = resolve_data_name(cfg)
    setting = f"{cfg.model_name}-{cfg.setting_strategy_label()}"
    experiment_path = build_experiment_path(cfg)
    eda_path = build_eda_path(cfg)
    data_root = Path(cfg.results_dir) / data_name
    artifacts = RunArtifacts(
        setting=setting,
        experiment_path=experiment_path,
        eda_path=eda_path,
        data_name=data_name,
        checkpoints_dir=data_root / "checkpoints" / experiment_path,
        train_results_dir=data_root / "results_train" / experiment_path,
        test_results_dir=data_root / "results_test" / experiment_path,
        forecast_results_dir=data_root / "results_forecast" / experiment_path,
        eda_dir=data_root / "results_eda" / eda_path,
        monitor_dir=data_root / "monitor" / experiment_path,
        custom_monitor_dir=data_root / "custom_monitor" / experiment_path,
    )
    # 创建结果目录：EDA-only 仅创建 EDA 目录，避免散落空模型目录与无意义实验路径。
    dirs = (
        [artifacts.eda_dir]
        if cfg.is_eda_only()
        else [
            artifacts.checkpoints_dir,
            artifacts.train_results_dir,
            artifacts.test_results_dir,
            artifacts.forecast_results_dir,
            artifacts.eda_dir,
            artifacts.monitor_dir,
            artifacts.custom_monitor_dir,
        ]
    )
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
    
    return artifacts
