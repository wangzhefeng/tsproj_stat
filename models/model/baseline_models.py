"""轻量统计基线模型。

手写基线（seasonal_naive/historic_average/croston）在此维护；
StatsForecast 后端基线（auto_ets/auto_theta/dynamic_theta/auto_ces/
random_walk_drift/seasonal_window_average）归 statsforecast_backend.py。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from models.contracts.inputs import preserve_univariate_series
from models.contracts.validation import validate_horizon


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
