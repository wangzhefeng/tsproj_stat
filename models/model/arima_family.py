from __future__ import annotations

import warnings
from itertools import product
from typing import Iterable

import pandas as pd

from models.base import BaseStatModel
from data_provider.data_transfer import to_univariate_series
from .fallbacks import (
    NaiveModel, TrendFallbackModel,
    FallbackMixin, validate_horizon, warn_and_use_fallback
)


def build_order_grid(p_values=(0, 1, 2), d_values=(0, 1), q_values=(0, 1, 2)):
    return [(p, d, q) for p, d, q in product(p_values, d_values, q_values)]


def select_arima_order(y: pd.Series, order_grid: Iterable[tuple[int, int, int]], ic: str = "aic"):
    if ic not in {"aic", "bic"}:
        raise ValueError("ic must be one of {'aic', 'bic'}")

    from statsmodels.tsa.arima.model import ARIMA

    best_order = None
    best_score = float("inf")

    for order in order_grid:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Non-invertible starting MA parameters found.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*Non-stationary starting autoregressive parameters found.*",
                    category=UserWarning,
                )
                result = ARIMA(y.astype(float), order=order).fit()
            score = float(getattr(result, ic))
            if score < best_score:
                best_score = score
                best_order = order
        except Exception:
            continue

    if best_order is None:
        raise RuntimeError("No valid ARIMA order found in order_grid")

    return best_order, best_score


class ARIMAModel(FallbackMixin, BaseStatModel):
    def __init__(self, order=(1, 1, 1), auto_order: bool = False, order_grid=None, ic: str = "aic"):
        self.order = order
        self.auto_order = auto_order
        self.order_grid = list(order_grid) if order_grid is not None else build_order_grid()
        self.ic = ic
        self.selected_order = order
        self.selected_score = None
        self._fallback = NaiveModel()
        self._result = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "ARIMAModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.arima.model import ARIMA

            fit_order = self.order
            if self.auto_order:
                fit_order, best_score = select_arima_order(series, self.order_grid, self.ic)
                self.selected_order = fit_order
                self.selected_score = best_score
            else:
                self.selected_order = self.order
                self.selected_score = None

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Non-invertible starting MA parameters found.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*Non-stationary starting autoregressive parameters found.*",
                    category=UserWarning,
                )
                self._result = ARIMA(series, order=fit_order).fit()
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="ARIMAModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        forecast = self._result.forecast(steps=horizon)
        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(forecast)
        return forecast.reset_index(drop=True).rename("yhat")


class SARIMAModel(FallbackMixin, BaseStatModel):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._fallback = TrendFallbackModel()
        self._result = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "SARIMAModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Non-invertible starting MA parameters found.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*Non-stationary starting autoregressive parameters found.*",
                    category=UserWarning,
                )
                self._result = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order).fit(disp=False)
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="SARIMAModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        return pd.Series(self._result.forecast(steps=horizon), name="yhat").reset_index(drop=True)


class AutoARIMAModel(FallbackMixin, BaseStatModel):
    def __init__(self, seasonal: bool = False, m: int = 1):
        self.seasonal = seasonal
        self.m = m
        self._result = None
        self._fallback = ARIMAModel(auto_order=True)

    def fit(self, y: pd.Series | pd.DataFrame) -> "AutoARIMAModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            import pmdarima as pm

            self._result = pm.auto_arima(
                series,
                seasonal=self.seasonal,
                m=self.m,
                suppress_warnings=True,
                error_action="ignore",
            )
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="AutoARIMAModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        return pd.Series(self._result.predict(n_periods=horizon), name="yhat")
