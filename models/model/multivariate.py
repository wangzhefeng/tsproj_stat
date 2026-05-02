from __future__ import annotations

import pandas as pd

from models.base import BaseStatModel
from data_provider.data_transfer import to_dataframe
from .fallbacks import (
    NaiveModel, TrendFallbackModel,
    FallbackMixin, validate_horizon, warn_and_use_fallback
)

class VARModel(FallbackMixin, BaseStatModel):
    def __init__(self, maxlags: int | None = None):
        self.maxlags = maxlags
        self._result = None
        self._frame = None
        self._fallback = TrendFallbackModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "VARModel":
        frame = to_dataframe(y).astype(float)
        self._frame = frame.copy()
        self._fallback.fit(frame.iloc[:, 0])
        try:
            from statsmodels.tsa.api import VAR

            self._result = VAR(frame).fit(maxlags=self.maxlags)
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="VARModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None or self._frame is None:
            return self._fallback_predict(horizon)
        lag = self._result.k_ar
        input_values = self._frame.values[-lag:]
        forecast = self._result.forecast(input_values, steps=horizon)
        return pd.Series(forecast[:, 0], name="yhat")


class BayesianVARModel(VARModel):
    pass


class LinearVARModel(VARModel):
    pass
