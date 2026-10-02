"""模型层的运行事实：由模型声明原生状态字段，产物层不猜后端。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.base import BaseStatModel


@dataclass(frozen=True)
class ModelRuntimeInfo:
    has_native_result: bool
    uses_fallback_model: bool
    fallback_model_class: str | None
    using_fallback_prediction: bool
    is_trainer_fallback: bool
    fallback_reason: str | None


def model_runtime_info(model: BaseStatModel) -> ModelRuntimeInfo:
    """声明字段均有效时才有原生状态；备用对象存在不等于正在降级。"""
    fields = model._runtime_backend_fields
    native = bool(fields) and all(getattr(model, field, None) is not None for field in fields)
    fallback = getattr(model, "_fallback", None)
    fallback_name = type(fallback).__name__ if fallback is not None else model._runtime_fallback_name
    trainer = model._is_fallback
    used = trainer or (fallback_name is not None and not native)
    reason = model._fallback_reason if used else None
    if used and reason is None:
        reason = "native fit unavailable or insufficient history"
    return ModelRuntimeInfo(native, fallback_name is not None, fallback_name, used, trainer, reason)
