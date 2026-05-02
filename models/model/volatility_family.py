from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from data_provider.data_transfer import to_univariate_series
from .fallbacks import (
    NaiveModel, TrendFallbackModel,
    FallbackMixin, validate_horizon, warn_and_use_fallback
)


class ARCHModel(FallbackMixin, BaseStatModel):
    def __init__(self):
        self._result = None
        self._fallback = NaiveModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "ARCHModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from arch import arch_model

            self._result = arch_model(series, mean="Constant", vol="ARCH", p=1).fit(disp="off")
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="ARCHModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        fc = self._result.forecast(horizon=horizon)
        values = np.asarray(fc.mean.iloc[-1]).reshape(-1)
        return pd.Series(values[:horizon], name="yhat")


class GARCHModel(FallbackMixin, BaseStatModel):
    def __init__(self):
        self._result = None
        self._fallback = NaiveModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "GARCHModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from arch import arch_model

            self._result = arch_model(series, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="GARCHModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        fc = self._result.forecast(horizon=horizon)
        values = np.asarray(fc.mean.iloc[-1]).reshape(-1)
        return pd.Series(values[:horizon], name="yhat")
