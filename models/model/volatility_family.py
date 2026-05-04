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
    """
    自回归条件异方差模型
    """
    def __init__(self):
        self._result = None
        self._fallback = NaiveModel()

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "ARCHModel":
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

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        fc = self._result.forecast(horizon=horizon)
        values = np.asarray(fc.mean.iloc[-1]).reshape(-1)
        return pd.Series(values[:horizon], name="yhat")

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        import pandas as pd, numpy as np
        from scipy import stats
        if self._result is None:
            return super().predict_with_intervals(horizon, X_future, alpha)
        try:
            fc = self._result.forecast(horizon=horizon)
            mean = np.asarray(fc.mean.iloc[-1]).reshape(-1)[:horizon]
            var = np.asarray(fc.variance.iloc[-1]).reshape(-1)[:horizon]
            z = stats.norm.ppf(1 - alpha / 2)
            std = np.sqrt(np.maximum(var, 0))
            return pd.DataFrame({
                "yhat": mean,
                "yhat_lower": mean - z * std,
                "yhat_upper": mean + z * std,
            })
        except Exception:
            return super().predict_with_intervals(horizon, X_future, alpha)


class GARCHModel(FallbackMixin, BaseStatModel):
    """
    广义自回归条件异方差模型(GARCH)
    """
    def __init__(self):
        self._result = None
        self._fallback = NaiveModel()

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "GARCHModel":
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

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        fc = self._result.forecast(horizon=horizon)
        values = np.asarray(fc.mean.iloc[-1]).reshape(-1)
        return pd.Series(values[:horizon], name="yhat")

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        from scipy import stats
        if self._result is None:
            return super().predict_with_intervals(horizon, X_future, alpha)
        try:
            fc = self._result.forecast(horizon=horizon)
            mean = np.asarray(fc.mean.iloc[-1]).reshape(-1)[:horizon]
            var = np.asarray(fc.variance.iloc[-1]).reshape(-1)[:horizon]
            z = stats.norm.ppf(1 - alpha / 2)
            std = np.sqrt(np.maximum(var, 0))
            return pd.DataFrame({
                "yhat": mean,
                "yhat_lower": mean - z * std,
                "yhat_upper": mean + z * std,
            })
        except Exception:
            return super().predict_with_intervals(horizon, X_future, alpha)
