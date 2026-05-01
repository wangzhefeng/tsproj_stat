from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import BaseStatModel

from .common import FallbackMixin, to_univariate_series, validate_horizon, warn_and_use_fallback
from .exponential_family import ETSModel
from .fallbacks import TrendFallbackModel


class TBATSModel(FallbackMixin, BaseStatModel):
    def __init__(self):
        self._result = None
        self._fallback = ETSModel(seasonal="add", seasonal_periods=7)

    def fit(self, y: pd.Series | pd.DataFrame) -> "TBATSModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from tbats import TBATS

            self._result = TBATS().fit(series.values)
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="TBATSModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        pred = self._result.forecast(steps=horizon)
        return pd.Series(np.asarray(pred).reshape(-1), name="yhat")


class ProphetModel(FallbackMixin, BaseStatModel):
    def __init__(self):
        self._model = None
        self._last_ds = None
        self._fallback = TrendFallbackModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "ProphetModel":
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            from prophet import Prophet

            self._model = Prophet()
            if isinstance(series.index, pd.DatetimeIndex):
                ds = series.index
            else:
                ds = pd.date_range("2000-01-01", periods=len(series), freq="D")
            frame = pd.DataFrame({"ds": ds, "y": series.values})
            self._model.fit(frame)
            self._last_ds = ds[-1]
        except Exception as exc:
            self._model = None
            warn_and_use_fallback(
                model_name="ProphetModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._model is None or self._last_ds is None:
            return self._fallback_predict(horizon)
        future = pd.DataFrame({"ds": pd.date_range(self._last_ds, periods=horizon + 1, freq="D")[1:]})
        pred = self._model.predict(future)
        return pd.Series(pred["yhat"].values, name="yhat")


class NeuralProphetModel(TrendFallbackModel):
    pass


class BayesianTMTModel(FallbackMixin, TrendFallbackModel):
    def __init__(self, lags: list[int] | None = None):
        super().__init__()
        self.lags = sorted(set(lags or [1, 2, 7]))
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("lags must be positive integers")
        self._model = None
        self._history: list[float] = []
        self._fallback = TrendFallbackModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "BayesianTMTModel":
        series = to_univariate_series(y).astype(float)
        self._history = series.tolist()
        self._fallback.fit(series)

        max_lag = max(self.lags)
        if len(series) <= max_lag + 2:
            self._model = None
            return self

        x_rows = []
        y_vals = []
        for idx in range(max_lag, len(series)):
            x_rows.append([series.iloc[idx - lag] for lag in self.lags])
            y_vals.append(series.iloc[idx])

        try:
            from sklearn.linear_model import BayesianRidge

            model = BayesianRidge()
            model.fit(np.asarray(x_rows, dtype=float), np.asarray(y_vals, dtype=float))
            self._model = model
        except Exception as exc:
            self._model = None
            warn_and_use_fallback(
                model_name="BayesianTMTModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        if self._model is None:
            return self._fallback_predict(horizon)

        history = list(self._history)
        preds: list[float] = []
        for _ in range(horizon):
            if len(history) < max(self.lags):
                next_val = float(self._fallback.predict(1).iloc[0])
            else:
                feat = np.asarray([[history[-lag] for lag in self.lags]], dtype=float)
                next_val = float(self._model.predict(feat)[0])
            preds.append(next_val)
            history.append(next_val)
        return pd.Series(preds, name="yhat")


class RARModel(TrendFallbackModel):
    def __init__(self, alpha: float = 0.2):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        super().__init__()
        self.alpha = alpha
        self._resid_result = None
        self._last_index = 0

    def fit(self, y: pd.Series | pd.DataFrame) -> "RARModel":
        series = to_univariate_series(y).astype(float)
        super().fit(series)
        self._last_index = len(series) - 1

        x = np.arange(len(series), dtype=float)
        baseline = self._coef * x + self._intercept
        resid = series.values - baseline

        lag = max(1, int(round(self.alpha * 10)))
        if len(resid) <= lag + 2:
            self._resid_result = None
            return self

        try:
            from statsmodels.tsa.ar_model import AutoReg

            self._resid_result = AutoReg(resid, lags=lag, old_names=False).fit()
        except Exception as exc:
            self._resid_result = None
            warn_and_use_fallback(
                model_name="RARModel",
                fallback_name="TrendFallbackModel",
                exc=exc,
            )
        return self

    def predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        x_future = np.arange(self._last_index + 1, self._last_index + 1 + horizon, dtype=float)
        baseline = self._coef * x_future + self._intercept
        if self._resid_result is None:
            return pd.Series(baseline, name="yhat")

        try:
            resid_fc = self._resid_result.forecast(steps=horizon)
            return pd.Series(baseline + np.asarray(resid_fc, dtype=float), name="yhat")
        except Exception:
            return pd.Series(baseline, name="yhat")
