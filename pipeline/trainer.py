"""训练阶段执行器：模型创建、统一 fit 契约调用与 NaiveModel fallback。"""
from __future__ import annotations

import pandas as pd
from typing import Callable

from models.factory import ModelFactory
from models.base import BaseStatModel
from models.model.fallbacks import NaiveModel
from forecasting.strategies import checked_model_builder

from utils.log_util import logger


class Trainer:
    """训练阶段薄封装。

    只负责通过 ModelFactory 创建模型并调用统一 fit 契约；训练失败时回退到 NaiveModel，
    让后续产物仍能明确记录 fallback 原因。
    """

    def __init__(self, model_name: str, model_params: dict | None = None, ignore_unsupported_inputs=False):
        """
        Args:
            model_name: registry 中的模型名。
            model_params: 覆盖 registry 默认参数的超参字典。
            ignore_unsupported_inputs: 显式容忍模型不支持的协变量输入（默认拒绝）。
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.factory = ModelFactory()
        self.ignore_unsupported_inputs = ignore_unsupported_inputs

    def train(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
        model_builder: Callable[[], BaseStatModel] | None = None,
    ) -> BaseStatModel:
        """拟合模型；异常时返回带 fallback 标记的 NaiveModel。"""
        # 门禁口径与 forecast 阶段的 direct 策略一致：训练期输入能力校验
        # （native 多步/未来外生/历史协变量）按 direct 语义检查，具体规则归
        # forecasting.strategies.checked_model_builder。
        build_model = checked_model_builder(
            model_builder or (lambda: self.factory.create_model(self.model_name, self.model_params, self.ignore_unsupported_inputs)),
            "direct", X_future, history=y, X_hist=X_hist,
        )
        model = build_model()
        try:
            model.fit(y=y, X_hist=X_hist, X_future=X_future)
            logger.info(f"[Train] {self.model_name} fit success (n={len(y)})")
            return model
        except ValueError:
            # 输入/配置契约失败不是优化器失败，不允许通过 fallback 隐藏。
            raise
        except Exception as exc:
            logger.warning(f"[Train] {self.model_name} fit failed: {exc}. Falling back to NaiveModel.")
            fallback = NaiveModel()
            fallback.fit(y)
            fallback._is_fallback = True
            fallback._fallback_reason = str(exc)
            return fallback
