from __future__ import annotations

import itertools

import pandas as pd

from data_provider.data_processor import infer_seasonal_period
from models.base import BaseStatModel
from data_provider.data_transfer import to_univariate_series
from .fallbacks import (
    NaiveModel, TrendFallbackModel,
    FallbackMixin, validate_horizon, warn_and_use_fallback
)

class ETSModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        tune_smoothing_params: bool = False,
        smoothing_grid_level: list[float] | None = None,
        smoothing_grid_trend: list[float] | None = None,
        smoothing_grid_seasonal: list[float] | None = None,
        validation_size: int | None = None,
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.tune_smoothing_params = tune_smoothing_params
        self.smoothing_grid_level = smoothing_grid_level
        self.smoothing_grid_trend = smoothing_grid_trend
        self.smoothing_grid_seasonal = smoothing_grid_seasonal
        self.validation_size = validation_size
        self._fallback = TrendFallbackModel()
        self._result = None
        self._resolved_seasonal_periods: int | None = seasonal_periods

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "ETSModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        self._validate_configuration()
        self._resolved_seasonal_periods = self._resolve_seasonal_periods(series)
        try:
            if self.tune_smoothing_params:
                self._result = self._fit_with_tuning(series)
            else:
                self._result = self._fit_model(series)
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="ETSModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        return pd.Series(self._result.forecast(horizon), name="yhat").reset_index(drop=True)

    def _validate_configuration(self) -> None:
        valid_component = {None, "add", "mul"}
        if self.trend not in valid_component:
            raise ValueError("trend must be one of {None, 'add', 'mul'}")
        if self.seasonal not in valid_component:
            raise ValueError("seasonal must be one of {None, 'add', 'mul'}")
        for field_name, values in {
            "smoothing_grid_level": self.smoothing_grid_level,
            "smoothing_grid_trend": self.smoothing_grid_trend,
            "smoothing_grid_seasonal": self.smoothing_grid_seasonal,
        }.items():
            if values is None:
                continue
            if len(values) == 0:
                raise ValueError(f"{field_name} must not be empty when provided")
            if not all(0.0 < float(value) <= 1.0 for value in values):
                raise ValueError(f"{field_name} values must be in (0, 1]")
        if self.validation_size is not None and self.validation_size <= 0:
            raise ValueError("validation_size must be > 0 when provided")

    def _resolve_seasonal_periods(self, series: pd.Series) -> int | None:
        if self.seasonal is None:
            return None
        if self.seasonal_periods is not None:
            return self.seasonal_periods
        inferred = infer_seasonal_period(series)
        if inferred is None:
            raise ValueError("seasonal_periods is required for seasonal ETS when no reliable period can be inferred")
        return inferred

    def _fit_model(
        self,
        series: pd.Series,
        smoothing_level: float | None = None,
        smoothing_trend: float | None = None,
        smoothing_seasonal: float | None = None,
        optimized: bool = True,
    ):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        return ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self._resolved_seasonal_periods,
        ).fit(
            smoothing_level=smoothing_level,
            smoothing_trend=smoothing_trend,
            smoothing_seasonal=smoothing_seasonal,
            optimized=optimized,
        )

    def _fit_with_tuning(self, series: pd.Series):
        validation_size = self._resolve_validation_size(series)
        if validation_size is None:
            return self._fit_model(series)

        train = series.iloc[:-validation_size].reset_index(drop=True)
        valid = series.iloc[-validation_size:].reset_index(drop=True)
        best_params: tuple[float | None, float | None, float | None] | None = None
        best_mae = float("inf")
        last_error: Exception | None = None
        for params in self._iter_smoothing_candidates():
            try:
                result = self._fit_model(
                    train,
                    smoothing_level=params[0],
                    smoothing_trend=params[1],
                    smoothing_seasonal=params[2],
                    optimized=False,
                )
                pred = pd.Series(result.forecast(len(valid)), dtype=float)
                mae = float((pred.reset_index(drop=True) - valid).abs().mean())
                if mae < best_mae:
                    best_mae = mae
                    best_params = params
            except Exception as exc:
                last_error = exc

        if best_params is None:
            if last_error is not None:
                raise last_error
            raise ValueError("No valid smoothing parameter candidates were available for ETS tuning")

        return self._fit_model(
            series,
            smoothing_level=best_params[0],
            smoothing_trend=best_params[1],
            smoothing_seasonal=best_params[2],
            optimized=False,
        )

    def _resolve_validation_size(self, series: pd.Series) -> int | None:
        if len(series) < 4:
            return None
        proposed = self.validation_size or max(2, min(12, len(series) // 4))
        upper_bound = len(series) - 2
        if upper_bound < 1:
            return None
        return min(proposed, upper_bound)

    def _iter_smoothing_candidates(self):
        level_grid = self.smoothing_grid_level or [0.2, 0.4, 0.6, 0.8]
        trend_grid = self.smoothing_grid_trend or [0.2, 0.4, 0.6, 0.8]
        seasonal_grid = self.smoothing_grid_seasonal or [0.2, 0.4, 0.6, 0.8]
        if self.trend is None and self.seasonal is None:
            for level in level_grid:
                yield (float(level), None, None)
            return
        if self.seasonal is None:
            for level, trend in itertools.product(level_grid, trend_grid):
                yield (float(level), float(trend), None)
            return
        for level, trend, seasonal in itertools.product(level_grid, trend_grid, seasonal_grid):
            trend_value = None if self.trend is None else float(trend)
            yield (float(level), trend_value, float(seasonal))


class ThetaModel(FallbackMixin, BaseStatModel):
    
    def __init__(self, period: int = 1):
        self.period = period
        self._fallback = TrendFallbackModel()
        self._result = None

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "ThetaModel":
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

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        return pd.Series(self._result.forecast(horizon), name="yhat").reset_index(drop=True)
