from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from data_provider.data_transfer import to_univariate_series


def validate_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")


def warn_and_use_fallback(*, model_name: str, fallback_name: str, exc: Exception) -> None:
    warnings.warn(f"{model_name} fit failed, fallback to {fallback_name}: {exc}", RuntimeWarning)


class FallbackMixin:
    _fallback: BaseStatModel

    def _fallback_predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        return self._fallback.predict(horizon)


class NaiveModel(BaseStatModel):
    def __init__(self):
        self.last_value: float | None = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "NaiveModel":
        series = to_univariate_series(y)
        if len(series) == 0:
            raise ValueError("Input series is empty")
        self.last_value = float(series.iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self.last_value is None:
            raise RuntimeError("Model is not fitted")
        validate_horizon(horizon)
        return pd.Series([self.last_value] * horizon, name="yhat")


class TrendFallbackModel(BaseStatModel):
    def __init__(self):
        self._coef = 0.0
        self._intercept = 0.0
        self._last_index = 0

    def fit(self, y: pd.Series | pd.DataFrame) -> "TrendFallbackModel":
        series = to_univariate_series(y).astype(float)
        x = np.arange(len(series), dtype=float)
        if len(series) < 2:
            self._coef = 0.0
            self._intercept = float(series.iloc[-1])
        else:
            self._coef, self._intercept = np.polyfit(x, series.values, deg=1)
        self._last_index = len(series) - 1
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        x_future = np.arange(self._last_index + 1, self._last_index + 1 + horizon, dtype=float)
        y_future = self._coef * x_future + self._intercept
        return pd.Series(y_future, name="yhat")
