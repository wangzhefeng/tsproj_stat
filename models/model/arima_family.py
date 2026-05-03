from __future__ import annotations

import warnings
from itertools import product
from typing import Iterable

import pandas as pd

from data_provider.data_transfer import to_univariate_series
from models.base import BaseStatModel
from .fallbacks import (
    FallbackMixin,
    NaiveModel,
    TrendFallbackModel,
    validate_horizon,
    warn_and_use_fallback,
)


def _normalize_order(order: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if len(order) != 3:
        raise ValueError("order must contain exactly three integers: (p, d, q)")
    normalized = tuple(int(value) for value in order)
    if any(value < 0 for value in normalized):
        raise ValueError("order values must be non-negative")
    return normalized


def _normalize_seasonal_order(order: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    if len(order) != 4:
        raise ValueError("seasonal_order must contain exactly four integers: (P, D, Q, m)")
    normalized = tuple(int(value) for value in order)
    if any(value < 0 for value in normalized[:3]):
        raise ValueError("seasonal_order values P, D, Q must be non-negative")
    if normalized[3] <= 1:
        raise ValueError("seasonal_order period m must be > 1")
    return normalized


def build_order_grid(
    p_values: Iterable[int] = (0, 1, 2),
    d_values: Iterable[int] = (0, 1),
    q_values: Iterable[int] = (0, 1, 2),
) -> list[tuple[int, int, int]]:
    return [(int(p), int(d), int(q)) for p, d, q in product(p_values, d_values, q_values)]


def select_arima_order(y: pd.Series, order_grid: Iterable[tuple[int, int, int]], ic: str = "aic"):
    if ic not in {"aic", "bic"}:
        raise ValueError("ic must be one of {'aic', 'bic'}")

    from statsmodels.tsa.arima.model import ARIMA

    best_order = None
    best_score = float("inf")

    for order in order_grid:
        normalized_order = _normalize_order(order)
        try:
            with _fit_warning_context():
                result = ARIMA(y.astype(float), order=normalized_order).fit()
            score = float(getattr(result, ic))
            if score < best_score:
                best_score = score
                best_order = normalized_order
        except Exception:
            continue

    if best_order is None:
        raise RuntimeError("No valid ARIMA order found in order_grid")

    return best_order, best_score


class _fit_warning_context:
    def __enter__(self):
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
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
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


class ARIMAModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        order: tuple[int, int, int] | list[int] = (1, 1, 1),
        auto_order: bool = False,
        order_grid: Iterable[tuple[int, int, int]] | None = None,
        ic: str = "aic",
    ):
        self.order = _normalize_order(order)
        self.auto_order = auto_order
        self.order_grid = list(order_grid) if order_grid is not None else build_order_grid()
        self.ic = ic
        self.selected_order = self.order
        self.selected_score = None
        self._fallback = NaiveModel()
        self._result = None

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "ARIMAModel":
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

            with _fit_warning_context():
                self._result = ARIMA(series, order=fit_order).fit()
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name=type(self).__name__,
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        forecast = self._result.forecast(steps=horizon)
        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(forecast)
        return forecast.reset_index(drop=True).rename("yhat")


class ARModel(ARIMAModel):
    def __init__(self, p: int = 1):
        if p < 0:
            raise ValueError("p must be non-negative")
        super().__init__(order=(p, 0, 0))
        self.p = p


class MAModel(ARIMAModel):
    def __init__(self, q: int = 1):
        if q < 0:
            raise ValueError("q must be non-negative")
        super().__init__(order=(0, 0, q))
        self.q = q


class ARMAModel(ARIMAModel):
    def __init__(self, p: int = 1, q: int = 1):
        if p < 0 or q < 0:
            raise ValueError("p and q must be non-negative")
        super().__init__(order=(p, 0, q))
        self.p = p
        self.q = q


class SARIMAModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        order: tuple[int, int, int] | list[int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] | list[int] = (1, 1, 1, 7),
        trend: str | None = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        simple_differencing: bool = False,
        fit_kwargs: dict | None = None,
    ):
        self.order = _normalize_order(order)
        self.seasonal_order = _normalize_seasonal_order(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.simple_differencing = simple_differencing
        self.fit_kwargs = {"disp": False}
        if fit_kwargs is not None:
            self.fit_kwargs.update(fit_kwargs)
        self._fallback = TrendFallbackModel()
        self._result = None

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "SARIMAModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            with _fit_warning_context():
                self._result = SARIMAX(
                    series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend=self.trend,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility,
                    simple_differencing=self.simple_differencing,
                ).fit(**self.fit_kwargs)
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="SARIMAModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        return pd.Series(self._result.forecast(steps=horizon), name="yhat").reset_index(drop=True)


class AutoARIMAModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        seasonal: bool = False,
        m: int = 1,
        stepwise: bool = True,
        start_p: int = 2,
        start_q: int = 2,
        max_p: int = 5,
        max_q: int = 5,
        max_order: int = 5,
        d: int | None = None,
        test: str = "kpss",
        maxiter: int = 50,
        information_criterion: str = "aic",
        trace: bool = False,
        error_action: str = "ignore",
        suppress_warnings: bool = True,
    ):
        self.seasonal = seasonal
        self.m = m
        self.stepwise = stepwise
        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.max_order = max_order
        self.d = d
        self.test = test
        self.maxiter = maxiter
        self.information_criterion = information_criterion
        self.trace = trace
        self.error_action = error_action
        self.suppress_warnings = suppress_warnings
        self._result = None
        self._fallback: ARIMAModel | None = None

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "AutoARIMAModel":
        series = to_univariate_series(y).astype(float)
        self._result = None
        try:
            import pmdarima as pm

            self._result = pm.auto_arima(
                series,
                seasonal=self.seasonal,
                m=self.m,
                stepwise=self.stepwise,
                start_p=self.start_p,
                start_q=self.start_q,
                max_p=self.max_p,
                max_q=self.max_q,
                max_order=self.max_order,
                d=self.d,
                test=self.test,
                maxiter=self.maxiter,
                information_criterion=self.information_criterion,
                trace=self.trace,
                error_action=self.error_action,
                suppress_warnings=self.suppress_warnings,
            )
        except Exception as exc:
            self._result = None
            self._ensure_fallback_fitted(series)
            warn_and_use_fallback(
                model_name="AutoARIMAModel",
                fallback_name=type(self._fallback).__name__ if self._fallback is not None else "ARIMAModel",
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            if self._fallback is None:
                raise RuntimeError("Model is not fitted")
            return self._fallback_predict(horizon)
        return pd.Series(self._result.predict(n_periods=horizon), name="yhat")

    def _ensure_fallback_fitted(self, series: pd.Series) -> None:
        if self._fallback is None:
            self._fallback = ARIMAModel(auto_order=True)
        self._fallback.fit(series)
