"""将模型层公开的运行事实组织为可追踪产物元数据。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any
from models.base import BaseStatModel
from models.registry import MODEL_REGISTRY


def model_info_payload(model: BaseStatModel, model_params: dict[str, Any],
                       model_name: str | None = None) -> dict[str, Any]:
    spec = MODEL_REGISTRY.get(model_name or "")
    payload: dict[str, Any] = {
        "model_class": type(model).__name__, "model_params": model_params,
        "model_name": model_name, "stability": spec.stability if spec else None,
        "is_optional": spec is not None and spec.stability == "optional",
        "is_experimental": spec is not None and spec.stability == "experimental",
        **asdict(model.runtime_info()),
    }
    for attr in ("order", "seasonal_order", "selected_order", "selected_score", "ic", "seasonal", "m"):
        if hasattr(model, attr):
            value = getattr(model, attr)
            payload[attr] = list(value) if isinstance(value, tuple) else value
    return payload


def interval_metadata_payload(metadata: dict) -> dict:
    """区间计算使用数值 level 键；产物协议显式转为 JSON 对象键。"""
    result = dict(metadata)
    if "calibration_ranks" in result:
        result["calibration_ranks"] = {str(level): rank for level, rank in result["calibration_ranks"].items()}
    return result
