from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import asdict, dataclass

import pandas as pd

from config import AppConfig


# ##############################
# 准备运行结果参数
# ##############################
@dataclass(frozen=True)
class RunArtifacts:
    setting: str
    data_name: str
    checkpoints_dir: Path
    train_results_dir: Path
    test_results_dir: Path
    forecast_results_dir: Path
    eda_dir: Path


def _resolve_data_name(data_path: str | None) -> str:
    if data_path is None:
        return "demo_series"
    return Path(data_path).stem


def _build_setting(model_name: str, data_name: str, pred_method: str) -> str:
    return f"{model_name}-{data_name}-{pred_method}"


def prepare_run_artifacts(cfg: AppConfig) -> RunArtifacts:
    # 提取数据名称
    data_name = _resolve_data_name(cfg.data_path)
    # 构建结果目录
    setting = _build_setting(cfg.model_name, data_name, cfg.pred_method)
    # 创建结果参数实例
    artifacts = RunArtifacts(
        setting=setting,
        data_name=data_name,
        checkpoints_dir=Path(cfg.checkpoints_dir) / setting,
        train_results_dir=Path(cfg.train_results_dir) / setting,
        test_results_dir=Path(cfg.test_results_dir) / setting,
        forecast_results_dir=Path(cfg.forecast_result_dir) / setting,
        eda_dir=Path(cfg.eda_output_dir) / setting,
    )
    # 创建特定模型、数据、预测方法结果目录
    for path in (
        artifacts.checkpoints_dir,
        artifacts.train_results_dir,
        artifacts.test_results_dir,
        artifacts.forecast_results_dir,
        artifacts.eda_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    
    return artifacts


# ##############################
# 
# ##############################
def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def dataframe_to_csv(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def model_info_payload(model, model_params: dict[str, Any]) -> dict[str, Any]:
    fallback = getattr(model, "_fallback", None)
    result = getattr(model, "_result", None)
    payload: dict[str, Any] = {
        "model_class": type(model).__name__,
        "model_params": model_params,
        "uses_fallback_model": fallback is not None,
        "fallback_model_class": type(fallback).__name__ if fallback is not None else None,
        "using_fallback_prediction": fallback is not None and result is None,
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
    if history_time is None or history_time.empty:
        return pd.Series([pd.NaT] * horizon, name="timestamp")

    try:
        future_index = pd.date_range(start=pd.to_datetime(history_time.iloc[-1]), periods=horizon + 1, freq=freq)[1:]
    except Exception:
        return pd.Series([pd.NaT] * horizon, name="timestamp")
    return pd.Series(future_index, name="timestamp")


def dataclass_to_dict(instance) -> dict[str, Any]:
    return asdict(instance)
