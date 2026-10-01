"""统计模型统一抽象与创建入口：BaseStatModel 契约、registry 与工厂。"""
from .base import BaseStatModel
from .factory import ModelFactory
from .registry import MODEL_REGISTRY, create_stat_model

__all__ = [
    "BaseStatModel",
    "ModelFactory",
    "MODEL_REGISTRY",
    "create_stat_model",
]
