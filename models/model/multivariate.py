from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from data_provider.data_transfer import combine_history_frame, to_dataframe
from .fallbacks import (
    TrendFallbackModel,
    FallbackMixin,
    validate_horizon,
    warn_and_use_fallback,
)


def _require_multivariate_frame(
    y: pd.Series | pd.DataFrame,
    X_hist: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    frame = combine_history_frame(y, X_hist).astype(float)
    if frame.shape[1] < 2:
        raise ValueError("Multivariate models require at least two input columns in y/X_hist")
    return frame.reset_index(drop=True), frame.columns[0]


class VARModel(FallbackMixin, BaseStatModel):
    def __init__(self, maxlags: int | None = None, ic: str | None = None):
        self.maxlags = maxlags
        self.ic = ic
        self._result = None
        self._frame: pd.DataFrame | None = None
        self._target_col: str | None = None
        self._fallback = TrendFallbackModel()

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "VARModel":
        frame, target_col = _require_multivariate_frame(y, X_hist)
        self._frame = frame.copy()
        self._target_col = target_col
        self._fallback.fit(frame[target_col])
        try:
            from statsmodels.tsa.api import VAR

            self._result = VAR(frame).fit(maxlags=self.maxlags, ic=self.ic)
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="VARModel",
                fallback_name=type(self._fallback).__name__,
                exc=RuntimeError(f"{exc}. VAR typically expects approximately stationary multivariate input."),
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None or self._frame is None or self._target_col is None:
            return self._fallback_predict(horizon)
        lag = max(int(self._result.k_ar), 1)
        input_values = self._frame.values[-lag:]
        forecast = self._result.forecast(input_values, steps=horizon)
        target_idx = list(self._frame.columns).index(self._target_col)
        return pd.Series(forecast[:, target_idx], name="yhat")


class BayesianVARModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        time_lags: list[int] | tuple[int, ...] = (1, 2),
        burn_iter: int = 200,
        gibbs_iter: int = 50,
        random_seed: int = 2026,
    ):
        self.time_lags = tuple(sorted({int(lag) for lag in time_lags}))
        if not self.time_lags or any(lag <= 0 for lag in self.time_lags):
            raise ValueError("time_lags must contain positive integers")
        self.burn_iter = burn_iter
        self.gibbs_iter = gibbs_iter
        self.random_seed = random_seed
        self._fallback = TrendFallbackModel()
        self._frame: pd.DataFrame | None = None
        self._target_col: str | None = None
        self._A: np.ndarray | None = None
        self._Sigma: np.ndarray | None = None

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "BayesianVARModel":
        frame, target_col = _require_multivariate_frame(y, X_hist)
        self._frame = frame.copy()
        self._target_col = target_col
        self._fallback.fit(frame[target_col])
        x = frame.to_numpy(dtype=float)
        hd = max(self.time_lags)
        t_len, n_vars = x.shape
        if t_len <= hd + 1:
            self._A = None
            self._Sigma = None
            return self

        z_mat = x[hd:t_len, :]
        q_mat = np.zeros((t_len - hd, n_vars * len(self.time_lags)))
        for idx, lag in enumerate(self.time_lags):
            q_mat[:, idx * n_vars : (idx + 1) * n_vars] = x[(hd - lag):(t_len - lag), :]

        rng = np.random.default_rng(self.random_seed)
        a_mat = np.zeros((n_vars * len(self.time_lags), n_vars))
        sigma = np.eye(n_vars)
        try:
            from scipy.stats import invwishart

            for _ in range(self.burn_iter):
                psi0 = np.eye(n_vars * len(self.time_lags)) + q_mat.T @ q_mat
                psi = np.linalg.inv(psi0)
                m_mat = psi @ q_mat.T @ z_mat
                s_mat = np.eye(n_vars) + z_mat.T @ z_mat - m_mat.T @ psi0 @ m_mat
                sigma = invwishart.rvs(df=n_vars + t_len - hd, scale=s_mat, random_state=rng)
                a_mat = self._matrix_normal_sample(m_mat, psi, sigma, rng)
            self._A = a_mat
            self._Sigma = sigma
        except Exception as exc:
            self._A = None
            self._Sigma = None
            warn_and_use_fallback(
                model_name="BayesianVARModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._frame is None or self._target_col is None or self._A is None:
            return self._fallback_predict(horizon)

        history = self._frame.to_numpy(dtype=float)
        n_vars = history.shape[1]
        preds: list[float] = []
        target_idx = list(self._frame.columns).index(self._target_col)
        for _ in range(horizon):
            stacked = np.concatenate([history[-lag, :] for lag in self.time_lags], axis=0)
            next_vec = stacked @ self._A
            preds.append(float(next_vec[target_idx]))
            history = np.vstack([history, next_vec.reshape(1, n_vars)])
        return pd.Series(preds, name="yhat")

    @staticmethod
    def _matrix_normal_sample(
        mean: np.ndarray,
        row_cov: np.ndarray,
        col_cov: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        z = rng.standard_normal(size=mean.shape)
        row_chol = np.linalg.cholesky(row_cov)
        col_chol = np.linalg.cholesky(col_cov)
        return mean + row_chol @ z @ col_chol.T


class LinearVARModel(FallbackMixin, BaseStatModel):
    def __init__(
        self,
        target_lags: list[int] | tuple[int, ...] = (1, 2, 3),
        feature_lags: list[int] | tuple[int, ...] = (0, 1),
        require_future_exog: bool = False,
    ):
        self.target_lags = tuple(sorted({int(lag) for lag in target_lags}))
        self.feature_lags = tuple(sorted({int(lag) for lag in feature_lags}))
        if not self.target_lags or any(lag <= 0 for lag in self.target_lags):
            raise ValueError("target_lags must contain positive integers")
        if any(lag < 0 for lag in self.feature_lags):
            raise ValueError("feature_lags must be non-negative integers")
        self.require_future_exog = require_future_exog
        self._fallback = TrendFallbackModel()
        self._model = None
        self._frame: pd.DataFrame | None = None
        self._target_col: str | None = None
        self._feature_cols: list[str] = []
        self._future_exog: pd.DataFrame | None = None

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "LinearVARModel":
        frame = combine_history_frame(y, X_hist).astype(float).reset_index(drop=True)
        self._frame = frame
        self._target_col = frame.columns[0]
        self._feature_cols = frame.columns[1:].tolist()
        self._future_exog = None if X_future is None else to_dataframe(X_future).astype(float).reset_index(drop=True)
        self._fallback.fit(frame[self._target_col])

        max_lag = max(max(self.target_lags), max(self.feature_lags, default=0))
        if len(frame) <= max_lag + 1:
            self._model = None
            return self

        x_rows: list[list[float]] = []
        y_vals: list[float] = []
        for idx in range(max_lag, len(frame)):
            x_rows.append(self._build_training_row(frame, idx))
            y_vals.append(float(frame.iloc[idx][self._target_col]))

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(np.asarray(x_rows, dtype=float), np.asarray(y_vals, dtype=float))
            self._model = model
        except Exception as exc:
            self._model = None
            warn_and_use_fallback(
                model_name="LinearVARModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._model is None or self._frame is None or self._target_col is None:
            return self._fallback_predict(horizon)

        future_exog = X_future if X_future is not None else self._future_exog
        if future_exog is not None:
            future_exog = to_dataframe(future_exog).astype(float).reset_index(drop=True)

        history = self._frame.copy()
        preds: list[float] = []
        for step_idx in range(horizon):
            feature_row = self._build_forecast_row(history, step_idx, future_exog)
            next_val = float(self._model.predict(np.asarray([feature_row], dtype=float))[0])
            preds.append(next_val)
            new_row = self._build_next_history_row(history, next_val, step_idx, future_exog)
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        return pd.Series(preds, name="yhat")

    def _build_training_row(self, frame: pd.DataFrame, idx: int) -> list[float]:
        row = [float(frame.iloc[idx - lag][self._target_col]) for lag in self.target_lags]
        for col in self._feature_cols:
            row.extend(float(frame.iloc[idx - lag][col]) for lag in self.feature_lags)
        return row

    def _build_forecast_row(
        self,
        history: pd.DataFrame,
        step_idx: int,
        future_exog: pd.DataFrame | None,
    ) -> list[float]:
        current_idx = len(history)
        row = [float(history.iloc[current_idx - lag][self._target_col]) for lag in self.target_lags]
        for col in self._feature_cols:
            for lag in self.feature_lags:
                row.append(self._resolve_feature_value(history, future_exog, col, current_idx - lag, step_idx))
        return row

    def _build_next_history_row(
        self,
        history: pd.DataFrame,
        next_val: float,
        step_idx: int,
        future_exog: pd.DataFrame | None,
    ) -> dict[str, float]:
        next_row = {self._target_col: next_val}
        next_abs_idx = len(history)
        for col in self._feature_cols:
            next_row[col] = self._resolve_feature_value(history, future_exog, col, next_abs_idx, step_idx)
        return next_row

    def _resolve_feature_value(
        self,
        history: pd.DataFrame,
        future_exog: pd.DataFrame | None,
        col: str,
        abs_idx: int,
        step_idx: int,
    ) -> float:
        hist_len = len(history)
        if abs_idx < hist_len:
            return float(history.iloc[abs_idx][col])
        future_idx = abs_idx - hist_len
        if future_exog is not None and col in future_exog.columns and future_idx < len(future_exog):
            return float(future_exog.iloc[future_idx][col])
        if self.require_future_exog:
            raise ValueError(f"Missing future exogenous value for column '{col}' at forecast step {step_idx + 1}")
        return float(history.iloc[-1][col])
