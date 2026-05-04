from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from .fallbacks import validate_horizon


def _preserve_univariate_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("Input dataframe is empty")
        series = y.iloc[:, 0].copy()
    else:
        series = y.copy()
    return series.astype(float)


def _resolve_freq(series: pd.Series, freq: str | None = None) -> str:
    if isinstance(series.index, pd.DatetimeIndex):
        inferred = series.index.freqstr or pd.infer_freq(series.index)
        if inferred:
            return inferred
    return freq or "D"


def _build_single_series_frame(series: pd.Series, freq: str | None = None) -> tuple[pd.DataFrame, str]:
    resolved_freq = _resolve_freq(series, freq)
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
    def __init__(self, season_length: int = 1):
        if season_length <= 0:
            raise ValueError("season_length must be > 0")
        self.season_length = season_length
        self._pattern: list[float] = []

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "SeasonalNaiveModel":
        series = _preserve_univariate_series(y).reset_index(drop=True)
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
    def __init__(self, window: int | None = None):
        if window is not None and window <= 0:
            raise ValueError("window must be > 0 when provided")
        self.window = window
        self._mean: float | None = None

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "HistoricAverageModel":
        series = _preserve_univariate_series(y).reset_index(drop=True)
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
    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._forecast: float = 0.0
        self._fitted = False

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "CrostonModel":
        series = _preserve_univariate_series(y).reset_index(drop=True)
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
    def __init__(self, season_length: int = 1, freq: str | None = None):
        if season_length <= 0:
            raise ValueError("season_length must be > 0")
        self.season_length = season_length
        self.freq = freq
        self._sf = None
        self._column_name: str | None = None

    @staticmethod
    def _import_statsforecast():
        from statsforecast import StatsForecast

        return StatsForecast

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "_StatsForecastModelBase":
        series = _preserve_univariate_series(y)
        frame, resolved_freq = _build_single_series_frame(series, self.freq)
        statsforecast_cls = self._import_statsforecast()
        model = self._build_model()
        self._sf = statsforecast_cls(models=[model], freq=resolved_freq, n_jobs=1)
        self._sf.fit(frame)
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._sf is None:
            raise RuntimeError("Model is not fitted")
        pred = self._sf.predict(horizon)
        value_cols = [col for col in pred.columns if col not in {"unique_id", "ds"}]
        if not value_cols:
            raise RuntimeError("StatsForecast prediction output missing value column")
        return pd.Series(pred[value_cols[0]].astype(float).to_list(), name="yhat")

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        if self._sf is None:
            return super().predict_with_intervals(horizon, X_future, alpha)
        try:
            level = int(round((1 - alpha) * 100))
            pred = self._sf.predict(horizon, level=[level])
            value_cols = [col for col in pred.columns if col not in {"unique_id", "ds"}]
            if not value_cols:
                return super().predict_with_intervals(horizon, X_future, alpha)
            yhat_col = value_cols[0]
            lo_col = next((c for c in pred.columns if c.endswith(f"-lo-{level}")), None)
            hi_col = next((c for c in pred.columns if c.endswith(f"-hi-{level}")), None)
            import numpy as np
            return pd.DataFrame({
                "yhat": pred[yhat_col].astype(float).to_list(),
                "yhat_lower": pred[lo_col].astype(float).to_list() if lo_col else [np.nan] * horizon,
                "yhat_upper": pred[hi_col].astype(float).to_list() if hi_col else [np.nan] * horizon,
            })
        except Exception:
            return super().predict_with_intervals(horizon, X_future, alpha)


class AutoETSModel(_StatsForecastModelBase):
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
