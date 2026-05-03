from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from .exponential_family import ETSModel
from .fallbacks import FallbackMixin, TrendFallbackModel, validate_horizon, warn_and_use_fallback


def _preserve_univariate_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("Input dataframe is empty")
        series = y.iloc[:, 0].copy()
    else:
        series = y.copy()
    name = series.name or "y"
    series.name = name
    return series.astype(float)


def _resolve_series_freq(series: pd.Series, fallback_freq: str | None) -> str:
    if isinstance(series.index, pd.DatetimeIndex):
        inferred = series.index.freqstr or pd.infer_freq(series.index)
        if inferred:
            return inferred
    return fallback_freq or "D"


def _build_datetime_index(series: pd.Series, fallback_freq: str | None) -> tuple[pd.DatetimeIndex, str]:
    resolved_freq = _resolve_series_freq(series, fallback_freq)
    if isinstance(series.index, pd.DatetimeIndex):
        return pd.DatetimeIndex(series.index), resolved_freq
    return pd.date_range("2000-01-01", periods=len(series), freq=resolved_freq), resolved_freq


def _extract_regressor_frame(
    series: pd.Series,
    X_hist: pd.DataFrame | None,
    requested: list[str] | None,
) -> pd.DataFrame:
    if X_hist is None:
        if requested:
            raise ValueError("X_hist is required when regressors are configured")
        return pd.DataFrame(index=range(len(series)))

    frame = X_hist.reset_index(drop=True).copy()
    if len(frame) != len(series):
        raise ValueError("X_hist must have the same number of rows as y")

    target_name = series.name or "y"
    candidate_cols = list(frame.columns)
    if target_name in candidate_cols:
        candidate_cols = [col for col in candidate_cols if col != target_name]
    elif candidate_cols:
        try:
            first_values = frame.iloc[:, 0].astype(float).to_numpy()
            series_values = series.reset_index(drop=True).astype(float).to_numpy()
            if first_values.shape == series_values.shape and np.allclose(first_values, series_values, equal_nan=True):
                candidate_cols = candidate_cols[1:]
        except Exception:
            pass

    if requested is None:
        selected_cols = candidate_cols
    else:
        missing = [col for col in requested if col not in candidate_cols and col not in frame.columns]
        if missing:
            raise ValueError(f"Missing requested regressors in X_hist: {missing}")
        selected_cols = requested

    if not selected_cols:
        return pd.DataFrame(index=range(len(series)))
    return frame[selected_cols].astype(float).reset_index(drop=True)


def _extract_future_regressors(
    X_future: pd.DataFrame | None,
    regressor_names: list[str],
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    if not regressor_names:
        return pd.DataFrame(index=range(horizon))
    if X_future is None:
        raise ValueError(f"{model_name} requires future regressors when regressors are configured")
    frame = X_future.reset_index(drop=True).copy()
    missing = [col for col in regressor_names if col not in frame.columns]
    if missing:
        raise ValueError(f"{model_name} is missing future regressors: {missing}")
    if len(frame) < horizon:
        raise ValueError(f"{model_name} requires at least {horizon} future regressor rows")
    return frame[regressor_names].astype(float).iloc[:horizon].reset_index(drop=True)


class TBATSModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        seasonal_periods: list[int] | tuple[int, ...] | None = None,
        use_box_cox: bool | None = None,
        use_trend: bool | None = None,
        use_damped_trend: bool | None = None,
        use_arma_errors: bool = True,
        show_warnings: bool = True,
        n_jobs: int | None = None,
    ):
        self.seasonal_periods = list(seasonal_periods) if seasonal_periods is not None else None
        self.use_box_cox = use_box_cox
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.use_arma_errors = use_arma_errors
        self.show_warnings = show_warnings
        self.n_jobs = n_jobs
        fallback_period = self.seasonal_periods[0] if self.seasonal_periods else 7
        self._result = None
        self._fallback = ETSModel(seasonal="add", seasonal_periods=fallback_period)

    @staticmethod
    def _import_tbats():
        from tbats import TBATS

        return TBATS

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "TBATSModel":
        series = _preserve_univariate_series(y)
        self._fallback.fit(series.reset_index(drop=True))
        try:
            tbats_cls = self._import_tbats()
            model = tbats_cls(
                seasonal_periods=self.seasonal_periods,
                use_box_cox=self.use_box_cox,
                use_trend=self.use_trend,
                use_damped_trend=self.use_damped_trend,
                use_arma_errors=self.use_arma_errors,
                show_warnings=self.show_warnings,
                n_jobs=self.n_jobs,
            )
            self._result = model.fit(series.to_numpy(dtype=float))
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="TBATSModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        pred = self._result.forecast(steps=horizon)
        return pd.Series(np.asarray(pred, dtype=float).reshape(-1), name="yhat")


class ProphetModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "additive",
        country_holidays: str | None = None,
        yearly_seasonality: str | bool = "auto",
        weekly_seasonality: str | bool = "auto",
        daily_seasonality: str | bool = "auto",
        freq: str | None = None,
        regressors: list[str] | None = None,
    ):
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.country_holidays = country_holidays
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.freq = freq
        self.regressors = list(regressors) if regressors is not None else None
        self._model = None
        self._last_ds = None
        self._freq = freq or "D"
        self._regressor_names: list[str] = []
        self._fallback = TrendFallbackModel()

    @staticmethod
    def _import_prophet():
        from prophet import Prophet

        return Prophet

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "ProphetModel":
        series = _preserve_univariate_series(y)
        self._fallback.fit(series.reset_index(drop=True))
        try:
            ds, self._freq = _build_datetime_index(series, self.freq)
            reg_frame = _extract_regressor_frame(series, X_hist, self.regressors)
            self._regressor_names = list(reg_frame.columns)
            prophet_cls = self._import_prophet()
            self._model = prophet_cls(
                growth=self.growth,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
            )
            if self.country_holidays:
                self._model.add_country_holidays(country_name=self.country_holidays)
            for name in self._regressor_names:
                self._model.add_regressor(name)
            frame = pd.DataFrame({"ds": ds, "y": series.to_numpy(dtype=float)})
            for col in self._regressor_names:
                frame[col] = reg_frame[col].values
            self._model.fit(frame)
            self._last_ds = ds[-1]
        except Exception as exc:
            self._model = None
            self._regressor_names = []
            warn_and_use_fallback(
                model_name="ProphetModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._model is None or self._last_ds is None:
            return self._fallback_predict(horizon)
        future_ds = pd.date_range(self._last_ds, periods=horizon + 1, freq=self._freq)[1:]
        future = pd.DataFrame({"ds": future_ds})
        future_regs = _extract_future_regressors(X_future, self._regressor_names, horizon, "ProphetModel")
        for col in self._regressor_names:
            future[col] = future_regs[col].values
        pred = self._model.predict(future)
        return pd.Series(pred["yhat"].astype(float).to_list(), name="yhat")


class NeuralProphetModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        freq: str | None = None,
        yearly_seasonality: str | bool = "auto",
        weekly_seasonality: str | bool = "auto",
        daily_seasonality: str | bool = "auto",
        regressors: list[str] | None = None,
    ):
        self.freq = freq
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.regressors = list(regressors) if regressors is not None else None
        self._model = None
        self._freq = freq or "D"
        self._regressor_names: list[str] = []
        self._train_frame: pd.DataFrame | None = None
        self._fallback = TrendFallbackModel()

    @staticmethod
    def _import_neuralprophet():
        from neuralprophet import NeuralProphet

        return NeuralProphet

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "NeuralProphetModel":
        series = _preserve_univariate_series(y)
        self._fallback.fit(series.reset_index(drop=True))
        try:
            ds, self._freq = _build_datetime_index(series, self.freq)
            reg_frame = _extract_regressor_frame(series, X_hist, self.regressors)
            self._regressor_names = list(reg_frame.columns)
            neural_cls = self._import_neuralprophet()
            self._model = neural_cls(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                n_forecasts=1,
            )
            for name in self._regressor_names:
                self._model.add_future_regressor(name)
            frame = pd.DataFrame({"ds": ds, "y": series.to_numpy(dtype=float)})
            for col in self._regressor_names:
                frame[col] = reg_frame[col].values
            self._train_frame = frame
            self._model.fit(frame, freq=self._freq, minimal=True)
        except Exception as exc:
            self._model = None
            self._train_frame = None
            self._regressor_names = []
            warn_and_use_fallback(
                model_name="NeuralProphetModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._model is None or self._train_frame is None:
            return self._fallback_predict(horizon)
        future_regs = _extract_future_regressors(X_future, self._regressor_names, horizon, "NeuralProphetModel")
        future = self._model.make_future_dataframe(
            df=self._train_frame,
            periods=horizon,
            regressors_df=future_regs if not future_regs.empty else None,
        )
        pred = self._model.predict(future)
        if "yhat1" in pred.columns:
            values = pred["yhat1"].iloc[-horizon:]
        elif "yhat" in pred.columns:
            values = pred["yhat"].iloc[-horizon:]
        else:
            raise RuntimeError("NeuralProphet prediction output missing yhat column")
        return pd.Series(values.astype(float).to_list(), name="yhat")


class BayesianTMTModel(FallbackMixin, TrendFallbackModel):
    """
    Experimental single-series Bayesian lag-regression approximation.

    This is intentionally not the old BayesianTMF/BTMF panel-matrix algorithm from
    `models/models_todo/BayesianTMT.py`. The original method couples matrix
    factorization, imputation, and multivariate forecasting, which does not match the
    current single-target `fit/predict` contract.
    """

    def __init__(self, lags: list[int] | None = None):
        super().__init__()
        self.lags = sorted(set(lags or [1, 2, 7]))
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("lags must be positive integers")
        self._model = None
        self._history: list[float] = []
        self._fallback = TrendFallbackModel()

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "BayesianTMTModel":
        series = _preserve_univariate_series(y).reset_index(drop=True)
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

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
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
    """
    残差自回归模型(RAR)
    """

    def __init__(self, alpha: float = 0.2):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        super().__init__()
        self.alpha = alpha
        self._resid_result = None
        self._last_index = 0

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "RARModel":
        series = _preserve_univariate_series(y).reset_index(drop=True)
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

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
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
