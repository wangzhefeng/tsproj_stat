from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from dataclasses import asdict, dataclass

import pandas as pd

from config import AppConfig
from models.registry import MODEL_REGISTRY


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


def _resolve_data_name(data_path: str | None) -> str:
    """从数据路径提取数据名；demo 数据使用固定名称便于结果归类。"""
    if data_path is None:
        return "demo_series"
    return Path(data_path).stem


def _mapping_tokens(mapping: dict, prefix: str = "") -> list[str]:
    tokens: list[str] = []
    for key in sorted(mapping):
        full_key = f"{prefix}.{key}" if prefix else str(key)
        value = mapping[key]
        if isinstance(value, dict):
            tokens.extend(_mapping_tokens(value, full_key))
        else:
            tokens.append(f"{_path_token(full_key)}_{_path_token(value)}")
    return tokens


def _path_token(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, Path):
        value = value.name
    if isinstance(value, (list, tuple)):
        return "+".join(_path_token(item) for item in value) or "none"
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
    params = _path_token(resolved_params) if resolved_params else "default"
    scale = cfg.scaler_type if cfg.scale else "off"
    process = (
        f"process-denoise-{_path_token(cfg.denoise_method)}_w-{cfg.denoise_window}"
        f"_detrend-{_path_token(cfg.detrend_method)}"
        f"_decomp-{_path_token(cfg.decomposition_method)}"
        f"_period-{_path_token(cfg.seasonal_period)}"
    )
    return Path(
        f"{_path_token(cfg.model_name)}-{_path_token(cfg.setting_strategy_label())}",
        f"params-{params}",
        f"hist-{cfg.history_size}_pred-{cfg.predict_horizon}",
        (
            f"bt-{_path_token(cfg.resolved_backtest_window_mode())}"
            f"_train-{cfg.resolved_backtest_train_size()}"
            f"_h-{cfg.backtest_horizon}_step-{cfg.backtest_step}"
        ),
        (
            f"input-feature-{_path_token(cfg.feature_mode)}"
            f"_lags-{_path_token(cfg.lags)}_scale-{_path_token(scale)}"
        ),
        process,
        f"interval-{'on' if cfg.return_intervals else 'off'}_alpha-{_path_token(cfg.interval_alpha)}",
    )


def build_eda_path(cfg: AppConfig) -> Path:
    """构建只依赖数据准备与 EDA 参数的相对路径。"""
    aggregation = cfg.aggregation_method if cfg.aggregation_enabled else "none"
    fill_method = cfg.aggregation_fill_method if cfg.aggregation_enabled else "none"
    return Path(
        f"freq-{_path_token(cfg.freq)}",
        f"period-{cfg.eda_period}_nlags-{cfg.eda_nlags}",
        (
            f"recommend-{'on' if cfg.eda_recommendation_enabled else 'off'}"
            f"_preprocessed-{'on' if cfg.eda_run_preprocessed else 'off'}"
        ),
        f"aggregation-{_path_token(aggregation)}_fill-{_path_token(fill_method)}",
    )


def prepare_run_artifacts(cfg: AppConfig) -> RunArtifacts:
    """按 data_name 和完整参数路径创建本次运行的产物目录。"""
    # 提取数据名称
    data_name = _resolve_data_name(cfg.data_path)
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


# ##############################
# 模型运行结果保存
# ##############################
def write_json(path: Path, payload: dict[str, Any]) -> str:
    """写 JSON 并返回路径字符串，供 run_summary 汇总引用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def dataframe_to_csv(path: Path, df: pd.DataFrame) -> str:
    """写 CSV 并返回路径字符串，统一各阶段落盘行为。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def model_info_payload(model, model_params: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    """提取模型可追踪信息，包括 fallback 状态和常见阶数/评分字段。"""
    fallback = getattr(model, "_fallback", None)
    result = getattr(model, "_result", None)
    spec = MODEL_REGISTRY.get(model_name or "")
    payload: dict[str, Any] = {
        "model_class": type(model).__name__,
        "model_params": model_params,
        "model_name": model_name,
        "stability": spec.stability if spec is not None else None,
        "is_optional": spec.stability == "optional" if spec is not None else False,
        "is_experimental": spec.stability == "experimental" if spec is not None else False,
        "uses_fallback_model": fallback is not None,
        "fallback_model_class": type(fallback).__name__ if fallback is not None else None,
        "using_fallback_prediction": fallback is not None and result is None,
        "is_trainer_fallback": bool(getattr(model, "_is_fallback", False)),
        "fallback_reason": getattr(model, "_fallback_reason", None),
        "has_native_result": result is not None,
    }
    for attr in ("order", "seasonal_order", "selected_order", "selected_score", "ic", "seasonal", "m"):
        if hasattr(model, attr):
            value = getattr(model, attr)
            if isinstance(value, tuple):
                payload[attr] = list(value)
            else:
                payload[attr] = value
    return payload


def forecast_timestamps(history_time: pd.Series | None, horizon: int, freq: str) -> pd.Series:
    """根据历史最后一个时间戳生成未来预测时间索引。"""
    if history_time is None or history_time.empty:
        return pd.Series([pd.NaT] * horizon, name="timestamp")

    try:
        future_index = pd.date_range(start=pd.to_datetime(history_time.iloc[-1]), periods=horizon + 1, freq=freq)[1:]
    except Exception:
        return pd.Series([pd.NaT] * horizon, name="timestamp")
    return pd.Series(future_index, name="timestamp")


def dataclass_to_dict(instance) -> dict[str, Any]:
    return asdict(instance)
