"""轻量统计基线模型。

这些模型用于提供稳定 baseline 或低成本候选模型，不引入复杂训练流程。
部分 statsforecast 依赖模型会显式处理缺失依赖。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from models.contracts.inputs import preserve_univariate_series, resolve_series_freq
from models.contracts.validation import validate_horizon


def _build_single_series_frame(series: pd.Series, freq: str | None = None) -> tuple[pd.DataFrame, str]:
    resolved_freq = resolve_series_freq(series, freq)
    if isinstance(series.index, pd.DatetimeIndex):
        ds = pd.DatetimeIndex(series.index)
    else:
        ds = pd.date_range("2000-01-01", periods=len(series), freq=resolved_freq)
    frame = pd.DataFrame(
        {
            "unique_id": ["series_0"] * len(series),
            "ds": ds,
            "y": series.to_numpy(dtype=float),
        }
    )
    return frame, resolved_freq


class SeasonalNaiveModel(BaseStatModel):
    """季节朴素基线：预测值 = 上一季节周期的同位观测。"""
    def __init__(self, season_length: int = 1):
        if season_length <= 0:
            raise ValueError("season_length must be > 0")
        self.season_length = season_length
        self._pattern: list[float] = []

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "SeasonalNaiveModel":
        series = preserve_univariate_series(y).reset_index(drop=True)
        if len(series) == 0:
            raise ValueError("Input series is empty")
        take = min(self.season_length, len(series))
        self._pattern = series.iloc[-take:].astype(float).tolist()
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if not self._pattern:
            raise RuntimeError("Model is not fitted")
        repeated = [self._pattern[idx % len(self._pattern)] for idx in range(horizon)]
        return pd.Series(repeated, name="yhat")


class HistoricAverageModel(BaseStatModel):
    """历史均值基线：预测值 = 训练窗口（可选最近 window 行）的均值。"""
    def __init__(self, window: int | None = None):
        if window is not None and window <= 0:
            raise ValueError("window must be > 0 when provided")
        self.window = window
        self._mean: float | None = None

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "HistoricAverageModel":
        series = preserve_univariate_series(y).reset_index(drop=True)
        if len(series) == 0:
            raise ValueError("Input series is empty")
        values = series if self.window is None else series.iloc[-self.window :]
        self._mean = float(values.mean())
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._mean is None:
            raise RuntimeError("Model is not fitted")
        return pd.Series([self._mean] * horizon, name="yhat")


class CrostonModel(BaseStatModel):
    """Croston 间歇需求模型（experimental）：需求规模与间隔分别指数平滑。"""
    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._forecast: float = 0.0
        self._fitted = False

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "CrostonModel":
        series = preserve_univariate_series(y).reset_index(drop=True)
        if (series < 0).any():
            raise ValueError("CrostonModel requires non-negative demand values")
        if len(series) == 0:
            raise ValueError("Input series is empty")

        non_zero_idx = np.flatnonzero(series.to_numpy(dtype=float) > 0.0)
        if len(non_zero_idx) == 0:
            self._forecast = 0.0
            self._fitted = True
            return self

        z = float(series.iloc[non_zero_idx[0]])
        p = float(non_zero_idx[0] + 1)
        prev_idx = int(non_zero_idx[0])
        for idx in non_zero_idx[1:]:
            demand = float(series.iloc[idx])
            interval = float(idx - prev_idx)
            z = z + self.alpha * (demand - z)
            p = p + self.alpha * (interval - p)
            prev_idx = int(idx)
        self._forecast = 0.0 if p <= 0 else z / p
        self._fitted = True
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        return pd.Series([self._forecast] * horizon, name="yhat")


class _StatsForecastModelBase(BaseStatModel, ABC):
    """StatsForecast 后端模型的公共基类：单序列数组接口、fitted 值与区间协议。

    子类只需声明后端装配；pandas 3 CoW 下统一交付自有可写数组。
    """
    def __init__(self, season_length: int = 1, freq: str | None = None):
        if season_length <= 0:
            raise ValueError("season_length must be > 0")
        self.season_length = season_length
        self.freq = freq
        self._sf = None
        self._train_y: pd.Series | None = None
        self._column_name: str | None = None

    @staticmethod
    def _import_statsforecast():
        from statsforecast import StatsForecast

        return StatsForecast

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "_StatsForecastModelBase":
        series = preserve_univariate_series(y)
        # 单序列使用数组接口；freq 保留兼容，不再制造虚拟时间轴。
        # pandas 3 CoW 暴露只读视图，CES 内核会原地工作，必须交付自有可写数组。
        self._train_y = series.astype(float).reset_index(drop=True)
        self._sf = self._build_model().fit(series.to_numpy(dtype=float, copy=True))
        return self

    def fitted_values(self) -> pd.Series:
        # SF 2.0.1 forecast(fitted=True) 需重传训练序列。
        if self._sf is None or self._train_y is None:
            raise ValueError(
                f"{type(self).__name__} has no fitted result; fitted values unavailable"
            )
        fc = self._sf.forecast(self._train_y.to_numpy(dtype=float), 1, fitted=True)
        return pd.Series(np.asarray(fc["fitted"], dtype=float), name="fitted").reset_index(drop=True)

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._sf is None:
            raise RuntimeError("Model is not fitted")
        pred = self._sf.predict(horizon)
        return pd.Series(pred["mean"], dtype=float, name="yhat")

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        validate_horizon(horizon)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self._sf is None:
            raise RuntimeError("Model is not fitted")
        level = round((1 - alpha) * 100, 10)
        pred = self._sf.predict(horizon, level=[level])
        return pd.DataFrame({
            "yhat": pred["mean"],
            "yhat_lower": pred[f"lo-{level}"],
            "yhat_upper": pred[f"hi-{level}"],
        })

    def predict_with_levels(self, horizon: int, X_future=None, levels=None, alpha: float = 0.05):
        """SF 后端原生多水平：一次 predict(level=[...]) 返回全部水平列。"""
        from models.model.arima_family import statsforecast_levels_frame
        from forecasting.intervals import resolve_interval_levels

        validate_horizon(horizon)
        if self._sf is None:
            raise RuntimeError("Model is not fitted")
        resolved = resolve_interval_levels(levels, alpha)
        sf_levels = [round(level * 100, 10) for level in resolved]
        pred = self._sf.predict(horizon, level=sf_levels)
        return statsforecast_levels_frame(pred, resolved, len(resolved) > 1)


class AutoETSModel(_StatsForecastModelBase):
    """AutoETS（StatsForecast 后端）：自动选择指数平滑误差/趋势/季节形式。"""
    def __init__(
        self,
        season_length: int = 1,
        freq: str | None = None,
        model: str = "ZZZ",
        damped: bool | None = None,
        phi: float | None = None,
    ):
        super().__init__(season_length=season_length, freq=freq)
        self.model = model
        self.damped = damped
        self.phi = phi

    def _build_model(self):
        from statsforecast.models import AutoETS

        return AutoETS(
            season_length=self.season_length,
            model=self.model,
            damped=self.damped,
            phi=self.phi,
        )


class AutoThetaModel(_StatsForecastModelBase):
    """AutoTheta（StatsForecast 后端）：自动 Theta 变体选择。"""
    def __init__(
        self,
        season_length: int = 1,
        freq: str | None = None,
        decomposition_type: str = "multiplicative",
        model: str | None = None,
    ):
        super().__init__(season_length=season_length, freq=freq)
        self.decomposition_type = decomposition_type
        self.model = model

    def _build_model(self):
        from statsforecast.models import AutoTheta

        return AutoTheta(
            season_length=self.season_length,
            decomposition_type=self.decomposition_type,
            model=self.model,
        )


class DynamicThetaModel(_StatsForecastModelBase):
    """DynamicTheta（StatsForecast 后端）：动态优化 Theta。"""
    def __init__(
        self,
        season_length: int = 1,
        freq: str | None = None,
        decomposition_type: str = "multiplicative",
    ):
        super().__init__(season_length=season_length, freq=freq)
        self.decomposition_type = decomposition_type

    def _build_model(self):
        from statsforecast.models import DynamicTheta

        return DynamicTheta(
            season_length=self.season_length,
            decomposition_type=self.decomposition_type,
        )


class AutoCESModel(_StatsForecastModelBase):
    """AutoCES（StatsForecast 后端）：复杂指数平滑自动选择。"""
    def __init__(self, season_length: int = 1, model: str = "Z"):
        super().__init__(season_length)
        self.model = model

    def _build_model(self):
        from statsforecast.models import AutoCES
        return AutoCES(season_length=self.season_length, model=self.model)


class RandomWalkWithDriftModel(_StatsForecastModelBase):
    """带漂移随机游走（StatsForecast 后端）：预测值 = 末值 + 漂移×步长。"""
    def __init__(self):
        super().__init__()

    def _build_model(self):
        from statsforecast.models import RandomWalkWithDrift
        return RandomWalkWithDrift()


class SeasonalWindowAverageModel(_StatsForecastModelBase):
    """季节窗口均值（StatsForecast 后端）：同季位最近 window_size 期均值。"""
    def __init__(self, season_length: int = 7, window_size: int = 2):
        super().__init__(season_length)
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def _build_model(self):
        from statsforecast.models import SeasonalWindowAverage
        return SeasonalWindowAverage(season_length=self.season_length, window_size=self.window_size)
