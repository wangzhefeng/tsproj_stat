from __future__ import annotations

import json
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
    data_name: str
    checkpoints_dir: Path
    train_results_dir: Path
    test_results_dir: Path
    forecast_results_dir: Path
    eda_dir: Path


def _resolve_data_name(data_path: str | None) -> str:
    """从数据路径提取数据名；demo 数据使用固定名称便于结果归类。"""
    if data_path is None:
        return "demo_series"
    return Path(data_path).stem


def _build_setting(model_name: str, data_name: str, strategy_label: str) -> str:
    """统一构建 {model_name}-{data_name}-{strategy} 结果分组名。"""
    return f"{model_name}-{data_name}-{strategy_label}"


def prepare_run_artifacts(cfg: AppConfig) -> RunArtifacts:
    """为本次运行创建五类 setting 子目录。"""
    # 提取数据名称
    data_name = _resolve_data_name(cfg.data_path)
    # 构建结果目录
    setting = _build_setting(cfg.model_name, data_name, cfg.setting_strategy_label())
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
