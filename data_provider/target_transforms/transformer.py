"""目标变换：去噪、趋势、季节分解与缩放的统一状态管理。

纯算法（周期推断/去噪/分解）位于本包；本模块的 TargetTransformer
持有拟合状态（趋势斜率、季节模板等），负责 fit_transform 与预测值的逆变换重组。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.seasonality import infer_seasonal_period
from .denoising import remove_noise
from .decomposition import decompose, decompose_mstl
from .scaling import TargetScaler
from .trend import fit_trend


class TargetTransformer:
    """
    目标变换器；还原尺度和趋势/季节分量，不恢复去噪损失。

    主线支持轻量去噪、去趋势和季节分解。fit_transform() 会把目标序列转换到
    更适合统计模型学习的尺度；inverse_forecast() 再将预测值重组回原始业务尺度。
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
        seasonal_periods: list[int] | None = None,
        scale: bool = False,
        scaler_type: str = "standard",
    ):
        valid_detrend_methods = {"none", "linear", "moving_average"}
        valid_denoise_methods = {"none", "moving_average", "moving_median"}
        valid_decomposition_methods = {"none", "seasonal_decompose", "stl", "mstl"}
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
        if decomposition_method == "stl" and decomposition_model != "additive":
            raise ValueError("STL requires additive decomposition")
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
        self.seasonal_periods = sorted(seasonal_periods or [])
        if decomposition_method == "mstl":
            if (not self.seasonal_periods or len(set(self.seasonal_periods)) != len(self.seasonal_periods)
                    or any(isinstance(p, bool) or not isinstance(p, int) or p <= 1 for p in self.seasonal_periods)):
                raise ValueError("MSTL requires distinct integer seasonal_periods > 1")
            if seasonal_period is not None or decomposition_model != "additive":
                raise ValueError("MSTL requires additive decomposition and seasonal_periods only")
        elif self.seasonal_periods:
            raise ValueError("seasonal_periods requires MSTL")
        self._seasonal_templates: list[np.ndarray] = []
        self.decomposition_method = decomposition_method
        self.decomposition_target = decomposition_target
        self.decomposition_model = decomposition_model
        self.acf_max_lag = acf_max_lag
        self.seasonality_strength_threshold = seasonality_strength_threshold

        self.scaler = TargetScaler(scaler_type) if scale else None
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
        self.metadata: dict[str, str | int | None] = {}

    @property
    def enabled(self) -> bool:
        """任一预处理能力开启时，下游预测输出需要执行逆变换。"""
        return (
            self.scaler is not None
            or self.denoise_method != "none"
            or self.detrend_method != "none"
            or self.decomposition_method != "none"
        )

    def fit_transform(self, series: pd.Series) -> pd.Series:
        """拟合预处理参数并返回建模用序列。"""
        self._fitted = False
        self._seasonal_templates = []
        self.metadata = {"requested_method": self.decomposition_method,
                         "resolved_method": "simple", "resolved_period": None,
                         "fallback_reason": None}
        values = pd.Series(series).astype(float).reset_index(drop=True)
        if values.empty or not np.isfinite(values.to_numpy()).all():
            raise ValueError("target transform requires finite non-empty history")

        if self.denoise_method != "none":
            values = self.remove_noise(values, method=self.denoise_method, window=self.denoise_window)

        if self.decomposition_method != "none":
            transformed = self._fit_decomposition(values)
        else:
            transformed = self._fit_simple_transform(values)

        if self.scaler is not None:
            transformed = self.scaler.fit_transform(transformed)
        self._index_offset = len(values)
        self._fitted = True
        transformed.name = series.name
        return transformed

    def inverse_transform(self, transformed_series: pd.Series) -> pd.Series:
        """将训练期转换序列还原，主要用于验证可逆性。"""
        self._check_fitted()
        values = pd.Series(transformed_series).astype(float).reset_index(drop=True)
        if self.scaler is not None:
            values = self.scaler.inverse_transform(values)
        if self._mode == "decomposition":
            restored = self._inverse_from_components(values)
        else:
            trend = self._trend_for_length(len(values))
            restored = values + trend
        restored.name = transformed_series.name
        return restored

    def inverse_forecast(self, forecast_values: pd.Series | np.ndarray | list[float]) -> pd.Series:
        """将未来预测值从建模尺度还原到原始目标尺度。"""
        self._check_fitted()
        pred = pd.Series(forecast_values).astype(float).reset_index(drop=True)
        if self.scaler is not None:
            pred = self.scaler.inverse_transform(pred)
        if self._mode == "decomposition":
            return self._inverse_forecast_from_components(pred).rename("yhat")
        trend_future = self._future_trend(len(pred))
        return (pred + trend_future).rename("yhat")

    # 保留公开静态方法，算法只在本包计算模块中维护。
    remove_noise = staticmethod(remove_noise)

    def _fit_simple_transform(self, series: pd.Series) -> pd.Series:
        """无季节分解时只拟合趋势项，并返回去趋势后的序列。"""
        self._mode = "simple"
        trend = self._fit_trend(series)
        self._trend_train = trend
        self._seasonal_train = pd.Series(np.zeros(len(series)), index=series.index)
        self._seasonal_template = None
        self._resolved_period = None
        self._last_trend = float(trend.iloc[-1]) if len(trend) else 0.0
        return series - trend

    def _fit_decomposition(self, series: pd.Series) -> pd.Series:
        """拟合季节分解，并按 decomposition_target 决定模型学习目标。"""
        if self.decomposition_method == "mstl":
            return self._fit_mstl(series)
        period = self.seasonal_period or infer_seasonal_period(
            series,
            acf_max_lag=self.acf_max_lag,
            seasonality_strength_threshold=self.seasonality_strength_threshold,
        )
        if period is None or period < 2 or len(series) < max(period * 2, period + 2):
            if self.seasonal_period is not None:
                raise ValueError("explicit decomposition requires two complete seasonal cycles")
            self.metadata["fallback_reason"] = (
                "seasonal_period_not_inferred" if period is None else "insufficient_history_for_inferred_period"
            )
            return self._fit_simple_transform(series)

        trend, seasonal = decompose(series, period, self.decomposition_method, self.decomposition_model)
        self._mode = "decomposition"
        self._resolved_period = period
        self.metadata.update(resolved_method=self.decomposition_method, resolved_period=period)
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

    def _fit_mstl(self, series: pd.Series) -> pd.Series:
        trend, components = decompose_mstl(series, self.seasonal_periods)
        self._seasonal_templates = [components[-period:, i].copy() for i, period in enumerate(self.seasonal_periods)]
        self._mode = "decomposition"
        self._resolved_period = max(self.seasonal_periods)
        self.metadata.update(resolved_method="mstl", resolved_period=self._resolved_period)
        self._seasonal_train = pd.Series(components.sum(axis=1))
        self._trend_train = trend
        self._last_trend = float(self._trend_train.iloc[-1])
        transformed = series.reset_index(drop=True) - self._seasonal_train
        if self.decomposition_target == "resid_only":
            transformed = transformed - self._trend_train
        return transformed

    def _fit_trend(self, series: pd.Series) -> pd.Series:
        trend, self._slope, self._intercept = fit_trend(series, self.detrend_method, self.denoise_window)
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
        tail = self._future_seasonal(length - len(self._seasonal_train)).to_numpy(dtype=float)
        base = self._seasonal_train.to_numpy(dtype=float)
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
        base = self._trend_train.to_numpy(dtype=float)
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
        if self._seasonal_templates:
            return pd.Series(sum(np.resize(template, horizon) for template in self._seasonal_templates))
        if self._seasonal_template is None or self._resolved_period is None:
            return pd.Series(np.zeros(horizon))
        repeats = int(np.ceil(horizon / self._resolved_period))
        values = np.tile(self._seasonal_template.to_numpy(dtype=float), repeats)[:horizon]
        return pd.Series(values)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TargetTransformer is not fitted")
