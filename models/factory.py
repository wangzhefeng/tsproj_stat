"""模型工厂：按名称从 registry 创建模型实例，应用层不直接 import 具体模型类。"""
from __future__ import annotations

from dataclasses import dataclass

from .base import BaseStatModel
from .registry import MODEL_REGISTRY, create_stat_model


@dataclass
class ModelFactory:
    """模型创建入口。

    应用层只依赖工厂，不直接 import 具体模型类；新增模型必须先注册到 registry。
    """
    
    def create_model(self, model_name: str, model_params: dict | None = None, ignore_unsupported_inputs: bool = False) -> BaseStatModel:
        """按名称创建模型，并交给 registry 合并默认参数与校验参数名。"""
        model = create_stat_model(model_name, model_params)
        model._ignore_unsupported_inputs = ignore_unsupported_inputs
        return model

    @staticmethod
    def list_models() -> list[str]:
        """返回当前 registry 支持的模型名，用于 CLI/测试/文档展示。"""
        return sorted(MODEL_REGISTRY.keys())
