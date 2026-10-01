"""落盘原语：JSON/CSV 写入、模型信息提取与结果序列化。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from models.registry import MODEL_REGISTRY


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
