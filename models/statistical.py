from __future__ import annotations

import warnings
from dataclasses import dataclass
import inspect
from difflib import get_close_matches
import numpy as np
import pandas as pd

from .base import BaseStatModel
from .selection import build_order_grid, select_arima_order


class NaiveModel(BaseStatModel):
    def __init__(self):
        self.last_value: float | None = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "NaiveModel":
        series = _to_univariate_series(y)
        if len(series) == 0:
            raise ValueError("Input series is empty")
        self.last_value = float(series.iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self.last_value is None:
            raise RuntimeError("Model is not fitted")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        return pd.Series([self.last_value] * horizon, name="yhat")


class TrendFallbackModel(BaseStatModel):
    """Simple trend model used as robust fallback for optional deps."""

    def __init__(self):
        self._coef = 0.0
        self._intercept = 0.0
        self._last_index = 0

    def fit(self, y: pd.Series | pd.DataFrame) -> "TrendFallbackModel":
        series = _to_univariate_series(y).astype(float)
        x = np.arange(len(series), dtype=float)
        if len(series) < 2:
            self._coef = 0.0
            self._intercept = float(series.iloc[-1])
        else:
            self._coef, self._intercept = np.polyfit(x, series.values, deg=1)
        self._last_index = len(series) - 1
        return self

    def predict(self, horizon: int) -> pd.Series:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        x_future = np.arange(self._last_index + 1, self._last_index + 1 + horizon, dtype=float)
        y_future = self._coef * x_future + self._intercept
        return pd.Series(y_future, name="yhat")


class ARIMAModel(BaseStatModel):
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
        series = _to_univariate_series(y)
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
            self._result = ARIMA(series.astype(float), order=fit_order).fit()
        except Exception as exc:
            self._result = None
            warnings.warn(f"ARIMA fit failed, fallback to NaiveModel: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self._result is None:
            return self._fallback.predict(horizon)
        forecast = self._result.forecast(steps=horizon)
        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(forecast)
        return forecast.reset_index(drop=True).rename("yhat")


class SARIMAModel(BaseStatModel):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._fallback = TrendFallbackModel()
        self._result = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "SARIMAModel":
        series = _to_univariate_series(y)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            self._result = SARIMAX(series.astype(float), order=self.order, seasonal_order=self.seasonal_order).fit(disp=False)
        except Exception as exc:
            self._result = None
            warnings.warn(f"SARIMA fit failed, fallback to trend: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None:
            return self._fallback.predict(horizon)
        return pd.Series(self._result.forecast(steps=horizon), name="yhat").reset_index(drop=True)


class ETSModel(BaseStatModel):
    def __init__(self, trend: str | None = "add", seasonal: str | None = None, seasonal_periods: int | None = None):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fallback = TrendFallbackModel()
        self._result = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "ETSModel":
        series = _to_univariate_series(y)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            self._result = ExponentialSmoothing(
                series.astype(float),
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            ).fit()
        except Exception as exc:
            self._result = None
            warnings.warn(f"ETS fit failed, fallback to trend: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None:
            return self._fallback.predict(horizon)
        return pd.Series(self._result.forecast(horizon), name="yhat").reset_index(drop=True)


class ThetaModel(BaseStatModel):
    def __init__(self, period: int = 1):
        self.period = period
        self._fallback = TrendFallbackModel()
        self._result = None

    def fit(self, y: pd.Series | pd.DataFrame) -> "ThetaModel":
        series = _to_univariate_series(y)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel as _ThetaModel

            self._result = _ThetaModel(series.astype(float), period=self.period).fit()
        except Exception as exc:
            self._result = None
            warnings.warn(f"Theta fit failed, fallback to trend: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None:
            return self._fallback.predict(horizon)
        return pd.Series(self._result.forecast(horizon), name="yhat").reset_index(drop=True)


class VARModel(BaseStatModel):
    def __init__(self, maxlags: int | None = None):
        self.maxlags = maxlags
        self._result = None
        self._frame = None
        self._fallback = TrendFallbackModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "VARModel":
        frame = _to_dataframe(y)
        self._frame = frame.copy()
        self._fallback.fit(frame.iloc[:, 0])
        try:
            from statsmodels.tsa.api import VAR

            self._result = VAR(frame.astype(float)).fit(maxlags=self.maxlags)
        except Exception as exc:
            self._result = None
            warnings.warn(f"VAR fit failed, fallback to trend: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None or self._frame is None:
            return self._fallback.predict(horizon)
        lag = self._result.k_ar
        input_values = self._frame.values[-lag:]
        forecast = self._result.forecast(input_values, steps=horizon)
        return pd.Series(forecast[:, 0], name="yhat")


class AutoARIMAModel(BaseStatModel):
    def __init__(self, seasonal: bool = False, m: int = 1):
        self.seasonal = seasonal
        self.m = m
        self._result = None
        self._fallback = ARIMAModel(auto_order=True)

    def fit(self, y: pd.Series | pd.DataFrame) -> "AutoARIMAModel":
        series = _to_univariate_series(y)
        self._fallback.fit(series)
        try:
            import pmdarima as pm

            self._result = pm.auto_arima(series.astype(float), seasonal=self.seasonal, m=self.m, suppress_warnings=True, error_action="ignore")
        except Exception as exc:
            self._result = None
            warnings.warn(f"AutoARIMA fit failed, fallback to ARIMA auto_order: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None:
            return self._fallback.predict(horizon)
        return pd.Series(self._result.predict(n_periods=horizon), name="yhat")


class ARCHModel(BaseStatModel):
    def __init__(self):
        self._result = None
        self._fallback = NaiveModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "ARCHModel":
        series = _to_univariate_series(y)
        self._fallback.fit(series)
        try:
            from arch import arch_model

            self._result = arch_model(series.astype(float), mean="Constant", vol="ARCH", p=1).fit(disp="off")
        except Exception as exc:
            self._result = None
            warnings.warn(f"ARCH fit failed, fallback to naive: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None:
            return self._fallback.predict(horizon)
        fc = self._result.forecast(horizon=horizon)
        values = np.asarray(fc.mean.iloc[-1]).reshape(-1)
        return pd.Series(values[:horizon], name="yhat")


class GARCHModel(BaseStatModel):
    def __init__(self):
        self._result = None
        self._fallback = NaiveModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "GARCHModel":
        series = _to_univariate_series(y)
        self._fallback.fit(series)
        try:
            from arch import arch_model

            self._result = arch_model(series.astype(float), mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
        except Exception as exc:
            self._result = None
            warnings.warn(f"GARCH fit failed, fallback to naive: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None:
            return self._fallback.predict(horizon)
        fc = self._result.forecast(horizon=horizon)
        values = np.asarray(fc.mean.iloc[-1]).reshape(-1)
        return pd.Series(values[:horizon], name="yhat")


class TBATSModel(BaseStatModel):
    def __init__(self):
        self._result = None
        self._fallback = ETSModel(seasonal="add", seasonal_periods=7)

    def fit(self, y: pd.Series | pd.DataFrame) -> "TBATSModel":
        series = _to_univariate_series(y)
        self._fallback.fit(series)
        try:
            from tbats import TBATS

            self._result = TBATS().fit(series.astype(float).values)
        except Exception as exc:
            self._result = None
            warnings.warn(f"TBATS fit failed, fallback to ETS: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._result is None:
            return self._fallback.predict(horizon)
        pred = self._result.forecast(steps=horizon)
        return pd.Series(np.asarray(pred).reshape(-1), name="yhat")


class ProphetModel(BaseStatModel):
    def __init__(self):
        self._model = None
        self._last_ds = None
        self._fallback = TrendFallbackModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "ProphetModel":
        series = _to_univariate_series(y)
        self._fallback.fit(series)
        try:
            from prophet import Prophet

            self._model = Prophet()
            if isinstance(series.index, pd.DatetimeIndex):
                ds = series.index
            else:
                ds = pd.date_range("2000-01-01", periods=len(series), freq="D")
            df = pd.DataFrame({"ds": ds, "y": series.values})
            self._model.fit(df)
            self._last_ds = ds[-1]
        except Exception as exc:
            self._model = None
            warnings.warn(f"Prophet fit failed, fallback to trend: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._model is None or self._last_ds is None:
            return self._fallback.predict(horizon)
        future = pd.DataFrame({"ds": pd.date_range(self._last_ds, periods=horizon + 1, freq="D")[1:]})
        pred = self._model.predict(future)
        return pd.Series(pred["yhat"].values, name="yhat")


class NeuralProphetModel(TrendFallbackModel):
    pass


class BayesianVARModel(VARModel):
    pass


class LinearVARModel(VARModel):
    pass


class BayesianTMTModel(TrendFallbackModel):
    def __init__(self, lags: list[int] | None = None):
        super().__init__()
        self.lags = sorted(set(lags or [1, 2, 7]))
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("lags must be positive integers")
        self._model = None
        self._history: list[float] = []
        self._fallback = TrendFallbackModel()

    def fit(self, y: pd.Series | pd.DataFrame) -> "BayesianTMTModel":
        series = _to_univariate_series(y).astype(float)
        self._history = series.tolist()
        self._fallback.fit(series)

        max_lag = max(self.lags)
        if len(series) <= max_lag + 2:
            self._model = None
            return self

        x_rows = []
        y_vals = []
        for t in range(max_lag, len(series)):
            x_rows.append([series.iloc[t - lag] for lag in self.lags])
            y_vals.append(series.iloc[t])

        try:
            from sklearn.linear_model import BayesianRidge

            model = BayesianRidge()
            model.fit(np.asarray(x_rows, dtype=float), np.asarray(y_vals, dtype=float))
            self._model = model
        except Exception as exc:
            self._model = None
            warnings.warn(f"BayesianTMT fit failed, fallback to trend: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self._model is None:
            return self._fallback.predict(horizon)

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
        series = _to_univariate_series(y).astype(float)
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
            warnings.warn(f"RAR residual AR fit failed, fallback to trend: {exc}", RuntimeWarning)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        x_future = np.arange(self._last_index + 1, self._last_index + 1 + horizon, dtype=float)
        baseline = self._coef * x_future + self._intercept
        if self._resid_result is None:
            return pd.Series(baseline, name="yhat")

        try:
            resid_fc = self._resid_result.forecast(steps=horizon)
            return pd.Series(baseline + np.asarray(resid_fc, dtype=float), name="yhat")
        except Exception:
            return pd.Series(baseline, name="yhat")


@dataclass
class ModelSpec:
    cls: type[BaseStatModel]
    default_params: dict


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "naive": ModelSpec(NaiveModel, {}),
    "arima": ModelSpec(ARIMAModel, {"order": (1, 1, 1)}),
    "auto_arima": ModelSpec(AutoARIMAModel, {}),
    "sarima": ModelSpec(SARIMAModel, {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)}),
    "ets": ModelSpec(ETSModel, {}),
    "theta": ModelSpec(ThetaModel, {}),
    "var": ModelSpec(VARModel, {}),
    "bayesian_var": ModelSpec(BayesianVARModel, {}),
    "linear_var": ModelSpec(LinearVARModel, {}),
    "arch": ModelSpec(ARCHModel, {}),
    "garch": ModelSpec(GARCHModel, {}),
    "tbats": ModelSpec(TBATSModel, {}),
    "prophet": ModelSpec(ProphetModel, {}),
    "neuralprophet": ModelSpec(NeuralProphetModel, {}),
    "bayesian_tmt": ModelSpec(BayesianTMTModel, {}),
    "rar": ModelSpec(RARModel, {}),
}


def create_stat_model(name: str, params: dict | None = None) -> BaseStatModel:
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        suggestion = get_close_matches(key, MODEL_REGISTRY.keys(), n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise ValueError(f"Unsupported model '{name}'.{hint} Supported models: {supported}")
    spec = MODEL_REGISTRY[key]
    merged = dict(spec.default_params)
    if params:
        merged.update(params)
    signature = inspect.signature(spec.cls.__init__)
    valid_names = {k for k in signature.parameters if k != "self"}
    unknown = sorted([k for k in merged if k not in valid_names])
    if unknown:
        raise ValueError(
            f"Invalid params for model '{key}': {unknown}. "
            f"Accepted params: {sorted(valid_names)}"
        )
    valid_kwargs = {k: v for k, v in merged.items() if k in valid_names}
    return spec.cls(**valid_kwargs)


def _to_univariate_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("Input dataframe is empty")
        return y.iloc[:, 0].reset_index(drop=True)
    return y.reset_index(drop=True)


def _to_dataframe(y: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        return y.reset_index(drop=True)
    return pd.DataFrame({"y": y.reset_index(drop=True)})





