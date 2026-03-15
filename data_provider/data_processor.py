from __future__ import annotations

import numpy as np
import pandas as pd


class DataProcessor:
    """Preprocessing for denoising and detrending with reversible transform."""

    def __init__(
        self,
        detrend_method: str = "none",
        denoise_enabled: bool = False,
        denoise_window: int = 3,
    ):
        valid_methods = {"none", "linear", "moving_average"}
        if detrend_method not in valid_methods:
            raise ValueError(f"detrend_method must be one of {sorted(valid_methods)}")
        if denoise_window < 1:
            raise ValueError("denoise_window must be >= 1")

        self.detrend_method = detrend_method
        self.denoise_enabled = denoise_enabled
        self.denoise_window = denoise_window

        self._fitted = False
        self._trend_train: pd.Series | None = None
        self._index_offset = 0
        self._slope = 0.0
        self._intercept = 0.0
        self._last_trend = 0.0

    @property
    def enabled(self) -> bool:
        return self.denoise_enabled or self.detrend_method != "none"

    def fit_transform(self, series: pd.Series) -> pd.Series:
        values = pd.Series(series).astype(float).reset_index(drop=True)

        if self.denoise_enabled:
            values = self.remove_noise(values, window=self.denoise_window)

        trend = self._fit_trend(values)
        self._trend_train = trend
        self._index_offset = len(values)
        self._last_trend = float(trend.iloc[-1]) if len(trend) else 0.0
        self._fitted = True

        transformed = values - trend
        return transformed.rename(series.name)

    def inverse_transform(self, transformed_series: pd.Series) -> pd.Series:
        self._check_fitted()
        values = pd.Series(transformed_series).astype(float).reset_index(drop=True)
        trend = self._trend_for_length(len(values))
        return (values + trend).rename(transformed_series.name)

    def inverse_forecast(self, forecast_values: pd.Series | np.ndarray | list[float]) -> pd.Series:
        self._check_fitted()
        pred = pd.Series(forecast_values).astype(float).reset_index(drop=True)
        trend_future = self._future_trend(len(pred))
        return (pred + trend_future).rename("yhat")

    @staticmethod
    def remove_noise(series: pd.Series, window: int = 3) -> pd.Series:
        if window < 1:
            raise ValueError("window must be >= 1")
        return series.rolling(window=window, min_periods=1).mean()

    def _fit_trend(self, series: pd.Series) -> pd.Series:
        if self.detrend_method == "none":
            return pd.Series(np.zeros(len(series)), index=series.index)

        x = np.arange(len(series), dtype=float)
        if self.detrend_method == "linear":
            if len(series) < 2:
                self._slope = 0.0
                self._intercept = float(series.iloc[-1]) if len(series) else 0.0
            else:
                self._slope, self._intercept = np.polyfit(x, series.values, deg=1)
            trend = self._slope * x + self._intercept
            return pd.Series(trend, index=series.index)

        # moving_average trend
        trend = series.rolling(window=self.denoise_window, min_periods=1).mean()
        return trend

    def _trend_for_length(self, length: int) -> pd.Series:
        if self._trend_train is None:
            return pd.Series(np.zeros(length))
        if length <= len(self._trend_train):
            return self._trend_train.iloc[:length].reset_index(drop=True)
        if self.detrend_method == "linear":
            x = np.arange(length, dtype=float)
            return pd.Series(self._slope * x + self._intercept)
        ext = np.full(length - len(self._trend_train), self._last_trend, dtype=float)
        base = self._trend_train.reset_index(drop=True).values
        return pd.Series(np.concatenate([base, ext]))

    def _future_trend(self, horizon: int) -> pd.Series:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.detrend_method == "none":
            return pd.Series(np.zeros(horizon))
        if self.detrend_method == "linear":
            x = np.arange(self._index_offset, self._index_offset + horizon, dtype=float)
            return pd.Series(self._slope * x + self._intercept)
        return pd.Series(np.full(horizon, self._last_trend, dtype=float))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("DataProcessor is not fitted")
