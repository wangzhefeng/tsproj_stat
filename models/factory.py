from __future__ import annotations

from dataclasses import dataclass

from .base import BaseStatModel
from .statistical import MODEL_REGISTRY, create_stat_model


@dataclass
class ModelFactory:
    def create_model(self, model_name: str, model_params: dict | None = None) -> BaseStatModel:
        return create_stat_model(model_name, model_params)

    @staticmethod
    def list_models() -> list[str]:
        return sorted(MODEL_REGISTRY.keys())
