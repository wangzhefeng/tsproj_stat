from __future__ import annotations

import numpy as np
import pandas as pd


def infer_seasonal_period(
    series: pd.Series,
    acf_max_lag: int = 48,
    seasonality_strength_threshold: float = 0.3,
) -> int | None:
    values = pd.Series(series).astype(float).reset_index(drop=True)
    if len(values) < 4:
        return None

    max_lag = min(acf_max_lag, max(2, len(values) // 2))
    if max_lag <= 2:
        return None

    try:
        from statsmodels.tsa.stattools import acf

        acf_values = acf(values, nlags=max_lag, fft=True)
        best_lag = None
        best_score = float("-inf")
        for lag in range(2, len(acf_values)):
            score = float(acf_values[lag])
            prev_score = float(acf_values[lag - 1]) if lag - 1 >= 0 else float("-inf")
            next_score = float(acf_values[lag + 1]) if lag + 1 < len(acf_values) else float("-inf")
            is_local_peak = score >= prev_score and score >= next_score
            if is_local_peak and score > best_score:
                best_score = score
                best_lag = lag
        if best_lag is not None and best_score >= seasonality_strength_threshold:
            return int(best_lag)
    except Exception:
        pass

    centered = values - values.mean()
    if np.allclose(centered.values, 0.0):
        return None

    fft_values = np.fft.rfft(centered.values)
    power = np.abs(fft_values) ** 2
    if len(power) <= 1:
        return None
    power[0] = 0.0
    peak_idx = int(np.argmax(power))
    if peak_idx <= 0:
        return None
    period = int(round(len(values) / peak_idx))
    if period < 2 or period > max_lag:
        return None
    if float(power[peak_idx]) / max(float(power.sum()), 1e-8) < seasonality_strength_threshold:
        return None
    return period


class DataProcessor:
    """
    Preprocessing for denoising, detrending, and reversible decomposition transforms.
    """

    def __init__(
        self,
        detrend_method: str = "none",
        denoise_enabled: bool = False,
        denoise_method: str = "none",
        denoise_window: int = 3,
        seasonal_period: int | None = None,
        decomposition_method: str = "none",
        decomposition_target: str = "trend_resid",
        decomposition_model: str = "additive",
        acf_max_lag: int = 48,
        seasonality_strength_threshold: float = 0.3,
    ):
        valid_detrend_methods = {"none", "linear", "moving_average"}
        valid_denoise_methods = {"none", "moving_average", "moving_median"}
        valid_decomposition_methods = {"none", "seasonal_decompose", "stl"}
        valid_targets = {"trend_resid", "resid_only"}
        valid_models = {"additive", "multiplicative"}

        if detrend_method not in valid_detrend_methods:
            raise ValueError(f"detrend_method must be one of {sorted(valid_detrend_methods)}")
        if denoise_method not in valid_denoise_methods:
            raise ValueError(f"denoise_method must be one of {sorted(valid_denoise_methods)}")
        if denoise_window < 1:
            raise ValueError("denoise_window must be >= 1")
        if seasonal_period is not None and seasonal_period <= 1:
            raise ValueError("seasonal_period must be > 1 when provided")
        if decomposition_method not in valid_decomposition_methods:
            raise ValueError(f"decomposition_method must be one of {sorted(valid_decomposition_methods)}")
        if decomposition_target not in valid_targets:
            raise ValueError(f"decomposition_target must be one of {sorted(valid_targets)}")
        if decomposition_model not in valid_models:
            raise ValueError(f"decomposition_model must be one of {sorted(valid_models)}")
        if acf_max_lag <= 1:
            raise ValueError("acf_max_lag must be > 1")
        if not 0.0 <= seasonality_strength_threshold <= 1.0:
            raise ValueError("seasonality_strength_threshold must be in [0, 1]")

        self.detrend_method = detrend_method
        self.denoise_method = denoise_method
        if denoise_enabled and denoise_method == "none":
            self.denoise_method = "moving_average"
        self.denoise_enabled = denoise_enabled or self.denoise_method != "none"
        self.denoise_window = denoise_window
        self.seasonal_period = seasonal_period
        self.decomposition_method = decomposition_method
        self.decomposition_target = decomposition_target
        self.decomposition_model = decomposition_model
        self.acf_max_lag = acf_max_lag
        self.seasonality_strength_threshold = seasonality_strength_threshold

        self._fitted = False
        self._trend_train: pd.Series | None = None
        self._seasonal_train: pd.Series | None = None
        self._seasonal_template: pd.Series | None = None
        self._resolved_period: int | None = None
        self._index_offset = 0
        self._slope = 0.0
        self._intercept = 0.0
        self._last_trend = 0.0
        self._mode = "simple"

    @property
    def enabled(self) -> bool:
        return (
            self.denoise_method != "none"
            or self.detrend_method != "none"
            or self.decomposition_method != "none"
        )

    def fit_transform(self, series: pd.Series) -> pd.Series:
        values = pd.Series(series).astype(float).reset_index(drop=True)

        if self.denoise_method != "none":
            values = self.remove_noise(values, method=self.denoise_method, window=self.denoise_window)

        if self.decomposition_method != "none":
            transformed = self._fit_decomposition(values)
        else:
            transformed = self._fit_simple_transform(values)

        self._index_offset = len(values)
        self._fitted = True
        return transformed.rename(series.name)

    def inverse_transform(self, transformed_series: pd.Series) -> pd.Series:
        self._check_fitted()
        values = pd.Series(transformed_series).astype(float).reset_index(drop=True)
        if self._mode == "decomposition":
            return self._inverse_from_components(values).rename(transformed_series.name)
        trend = self._trend_for_length(len(values))
        return (values + trend).rename(transformed_series.name)

    def inverse_forecast(self, forecast_values: pd.Series | np.ndarray | list[float]) -> pd.Series:
        self._check_fitted()
        pred = pd.Series(forecast_values).astype(float).reset_index(drop=True)
        if self._mode == "decomposition":
            return self._inverse_forecast_from_components(pred).rename("yhat")
        trend_future = self._future_trend(len(pred))
        return (pred + trend_future).rename("yhat")

    @staticmethod
    def remove_noise(series: pd.Series, method: str = "moving_average", window: int = 3) -> pd.Series:
        if window < 1:
            raise ValueError("window must be >= 1")
        if method == "moving_average":
            return series.rolling(window=window, min_periods=1).mean()
        if method == "moving_median":
            if len(series) < window:
                return series.copy()
            return series.rolling(window=window, min_periods=1, center=True).median().bfill().ffill()
        if method == "none":
            return series.copy()
        raise ValueError("method must be one of {'none', 'moving_average', 'moving_median'}")

    def _fit_simple_transform(self, series: pd.Series) -> pd.Series:
        self._mode = "simple"
        trend = self._fit_trend(series)
        self._trend_train = trend
        self._seasonal_train = pd.Series(np.zeros(len(series)), index=series.index)
        self._seasonal_template = None
        self._resolved_period = None
        self._last_trend = float(trend.iloc[-1]) if len(trend) else 0.0
        return series - trend

    def _fit_decomposition(self, series: pd.Series) -> pd.Series:
        period = self.seasonal_period or infer_seasonal_period(
            series,
            acf_max_lag=self.acf_max_lag,
            seasonality_strength_threshold=self.seasonality_strength_threshold,
        )
        if period is None or period < 2 or len(series) < max(period * 2, period + 2):
            return self._fit_simple_transform(series)

        trend, seasonal = self._decompose(series, period)
        self._mode = "decomposition"
        self._resolved_period = period
        self._trend_train = trend.reset_index(drop=True)
        self._seasonal_train = seasonal.reset_index(drop=True)
        self._last_trend = float(self._trend_train.iloc[-1]) if len(self._trend_train) else 0.0
        self._seasonal_template = self._seasonal_train.iloc[-period:].reset_index(drop=True)

        if self.decomposition_target == "trend_resid":
            if self.decomposition_model == "additive":
                return series.reset_index(drop=True) - self._seasonal_train
            seasonal_safe = self._seasonal_train.replace(0.0, 1.0)
            return series.reset_index(drop=True) / seasonal_safe

        if self.decomposition_model == "additive":
            return series.reset_index(drop=True) - self._seasonal_train - self._trend_train
        base = (self._seasonal_train * self._trend_train).replace(0.0, 1.0)
        return series.reset_index(drop=True) / base

    def _decompose(self, series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
        values = series.reset_index(drop=True)
        if self.decomposition_method == "stl":
            from statsmodels.tsa.seasonal import STL

            result = STL(values, period=period, robust=True).fit()
            trend = pd.Series(result.trend).interpolate(limit_direction="both")
            seasonal = pd.Series(result.seasonal).interpolate(limit_direction="both")
            return trend, seasonal

        from statsmodels.tsa.seasonal import seasonal_decompose

        result = seasonal_decompose(
            values,
            period=period,
            model=self.decomposition_model,
            extrapolate_trend="freq",
        )
        trend = pd.Series(result.trend).interpolate(limit_direction="both")
        seasonal = pd.Series(result.seasonal).interpolate(limit_direction="both")
        return trend, seasonal

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

        trend = series.rolling(window=self.denoise_window, min_periods=1).mean()
        return trend

    def _inverse_from_components(self, values: pd.Series) -> pd.Series:
        seasonal = self._seasonal_for_length(len(values))
        trend = self._trend_for_length(len(values))
        if self.decomposition_target == "trend_resid":
            if self.decomposition_model == "additive":
                return values + seasonal
            return values * seasonal.replace(0.0, 1.0)
        if self.decomposition_model == "additive":
            return values + seasonal + trend
        return values * (seasonal * trend).replace(0.0, 1.0)

    def _inverse_forecast_from_components(self, pred: pd.Series) -> pd.Series:
        seasonal_future = self._future_seasonal(len(pred))
        trend_future = self._future_trend(len(pred))
        if self.decomposition_target == "trend_resid":
            if self.decomposition_model == "additive":
                return pred + seasonal_future
            return pred * seasonal_future.replace(0.0, 1.0)
        if self.decomposition_model == "additive":
            return pred + trend_future + seasonal_future
        return pred * (trend_future * seasonal_future).replace(0.0, 1.0)

    def _seasonal_for_length(self, length: int) -> pd.Series:
        if self._seasonal_train is None or self._resolved_period is None:
            return pd.Series(np.zeros(length))
        if length <= len(self._seasonal_train):
            return self._seasonal_train.iloc[:length].reset_index(drop=True)
        repeats = int(np.ceil((length - len(self._seasonal_train)) / self._resolved_period))
        tail = np.tile(self._seasonal_template.values, repeats)[: length - len(self._seasonal_train)]
        base = self._seasonal_train.reset_index(drop=True).values
        return pd.Series(np.concatenate([base, tail]))

    def _trend_for_length(self, length: int) -> pd.Series:
        if self._trend_train is None:
            return pd.Series(np.zeros(length))
        if length <= len(self._trend_train):
            return self._trend_train.iloc[:length].reset_index(drop=True)
        if self._mode == "simple" and self.detrend_method == "linear":
            x = np.arange(length, dtype=float)
            return pd.Series(self._slope * x + self._intercept)
        ext = np.full(length - len(self._trend_train), self._last_trend, dtype=float)
        base = self._trend_train.reset_index(drop=True).values
        return pd.Series(np.concatenate([base, ext]))

    def _future_trend(self, horizon: int) -> pd.Series:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self._mode == "decomposition":
            return pd.Series(np.full(horizon, self._last_trend, dtype=float))
        if self.detrend_method == "none":
            return pd.Series(np.zeros(horizon))
        if self.detrend_method == "linear":
            x = np.arange(self._index_offset, self._index_offset + horizon, dtype=float)
            return pd.Series(self._slope * x + self._intercept)
        return pd.Series(np.full(horizon, self._last_trend, dtype=float))

    def _future_seasonal(self, horizon: int) -> pd.Series:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self._seasonal_template is None or self._resolved_period is None:
            return pd.Series(np.zeros(horizon))
        repeats = int(np.ceil(horizon / self._resolved_period))
        values = np.tile(self._seasonal_template.values, repeats)[:horizon]
        return pd.Series(values)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("DataProcessor is not fitted")
