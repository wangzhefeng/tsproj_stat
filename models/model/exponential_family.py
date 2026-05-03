from __future__ import annotations

import pandas as pd

from models.base import BaseStatModel
from data_provider.data_transfer import to_univariate_series
from .fallbacks import (
    NaiveModel, TrendFallbackModel,
    FallbackMixin, validate_horizon, warn_and_use_fallback
)

class ETSModel(FallbackMixin, BaseStatModel):
    def __init__(self, trend: str | None = "add", seasonal: str | None = None, seasonal_periods: int | None = None):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fallback = TrendFallbackModel()
        self._result = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "ETSModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            self._result = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            ).fit()
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="ETSModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        return pd.Series(self._result.forecast(horizon), name="yhat").reset_index(drop=True)


class ThetaModel(FallbackMixin, BaseStatModel):
    
    def __init__(self, period: int = 1):
        self.period = period
        self._fallback = TrendFallbackModel()
        self._result = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "ThetaModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel as _ThetaModel

            self._result = _ThetaModel(series, period=self.period).fit()
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="ThetaModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        return pd.Series(self._result.forecast(horizon), name="yhat").reset_index(drop=True)
