"""模型 fallback 实现。

复杂模型拟合失败时使用这些简单模型保证流程可继续，并在 model_info 中记录原因。
"""

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
    """为模型类提供统一 fallback predict 入口。"""
    _fallback: BaseStatModel

    def _fallback_predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        return self._fallback.predict(horizon)


class NaiveModel(BaseStatModel):
    """最后值延续模型，作为最稳健的预测兜底。"""
    def __init__(self):
        self.last_value: float | None = None

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "NaiveModel":
        series = to_univariate_series(y)
        if len(series) == 0:
            raise ValueError("Input series is empty")
        self.last_value = float(series.iloc[-1])
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        if self.last_value is None:
            raise RuntimeError("Model is not fitted")
        validate_horizon(horizon)
        return pd.Series([self.last_value] * horizon, name="yhat")


class TrendFallbackModel(BaseStatModel):
    """线性趋势外推兜底，适用于保留趋势信息的单变量序列。"""
    def __init__(self):
        self._coef = 0.0
        self._intercept = 0.0
        self._last_index = 0

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "TrendFallbackModel":
        series = to_univariate_series(y).astype(float)
        x = np.arange(len(series), dtype=float)
        if len(series) < 2:
            self._coef = 0.0
            self._intercept = float(series.iloc[-1])
        else:
            self._coef, self._intercept = np.polyfit(x, series.values, deg=1)
        self._last_index = len(series) - 1
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        x_future = np.arange(self._last_index + 1, self._last_index + 1 + horizon, dtype=float)
        y_future = self._coef * x_future + self._intercept
        return pd.Series(y_future, name="yhat")
