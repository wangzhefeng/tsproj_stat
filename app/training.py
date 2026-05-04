from __future__ import annotations

import pandas as pd

from models.factory import ModelFactory
from models.model.fallbacks import NaiveModel

import os
from pathlib import Path
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ.setdefault('LOG_NAME', LOGGING_LABEL)
from utils.log_util import logger


class Trainer:

    def __init__(self, model_name: str, model_params: dict | None = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.factory = ModelFactory()

    def train(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ):
        model = self.factory.create_model(self.model_name, self.model_params)
        try:
            model.fit(y=y, X_hist=X_hist, X_future=X_future)
            logger.info(f"[Train] {self.model_name} fit success (n={len(y)})")
            return model
        except Exception as exc:
            logger.warning(f"[Train] {self.model_name} fit failed: {exc}. Falling back to NaiveModel.")
            fallback = NaiveModel({})
            fallback.fit(y)
            fallback._is_fallback = True
            fallback._fallback_reason = str(exc)
            return fallback
