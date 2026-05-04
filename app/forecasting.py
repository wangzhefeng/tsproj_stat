from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from models.factory import ModelFactory
from data_provider.data_transfer import combine_history_frame, to_univariate_series

import os
from pathlib import Path
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ.setdefault('LOG_NAME', LOGGING_LABEL)
from utils.log_util import logger


def _validate_forecast(yhat: pd.Series, horizon: int, model_name: str) -> pd.Series:
    """验证预测输出的长度、NaN、inf，就地修复 NaN，对 inf 抛出异常。"""
    if len(yhat) != horizon:
        raise ValueError(
            f"[{model_name}] forecast length {len(yhat)} != horizon {horizon}"
        )
    nan_count = int(yhat.isna().sum())
    if nan_count > 0:
        logger.warning(
            f"[{model_name}] {nan_count}/{horizon} NaN in forecast, "
            "filling with forward/backward fill"
        )
        yhat = yhat.ffill().bfill()
    if np.isinf(yhat.values).any():
        raise ValueError(f"[{model_name}] inf values detected in forecast output")
    return yhat


def run_inference(
    model,
    history: pd.Series | pd.DataFrame,
    horizon: int,
    pred_method: str = "direct",
    X_hist: pd.DataFrame | None = None,
    X_future: pd.DataFrame | None = None,
    model_builder=None,
    model_name: str = "unknown",
) -> pd.Series:
    method = pred_method.lower()
    if method not in {"one_step", "recursive", "direct"}:
        raise ValueError("pred_method must be one of {'one_step','recursive','direct'}")

    if method == "one_step":
        model.fit(history, X_hist=X_hist, X_future=X_future)
        yhat = model.predict(1, X_future=X_future.iloc[:1].reset_index(drop=True) if X_future is not None else None)
        return _validate_forecast(yhat, 1, model_name)

    if method == "direct":
        model.fit(history, X_hist=X_hist, X_future=X_future)
        yhat = model.predict(horizon, X_future=X_future)
        return _validate_forecast(yhat, horizon, model_name)

    # recursive
    hist = to_univariate_series(history)
    hist_frame = combine_history_frame(history, X_hist)
    preds = []
    for _ in range(horizon):
        model_i = model_builder() if model_builder is not None else copy.deepcopy(model)
        next_future = None
        if X_future is not None:
            step_idx = len(preds)
            if step_idx >= len(X_future):
                raise ValueError(
                    "Recursive forecasting requires complete future exogenous rows for every forecast step"
                )
            next_future = X_future.iloc[step_idx : step_idx + 1].reset_index(drop=True)
        model_i.fit(hist, X_hist=hist_frame, X_future=next_future)
        next_val = float(model_i.predict(1, X_future=next_future).iloc[0])
        preds.append(next_val)
        hist = pd.concat([hist, pd.Series([next_val])], ignore_index=True)
        next_row = hist_frame.iloc[-1].copy()
        next_row.iloc[0] = next_val
        if next_future is not None:
            for col in next_future.columns:
                next_row[col] = next_future.iloc[0][col]
        hist_frame = pd.concat([hist_frame, pd.DataFrame([next_row])], ignore_index=True)

    yhat = pd.Series(preds, name="yhat")
    return _validate_forecast(yhat, horizon, model_name)


class Forecaster:

    def __init__(self, model_name: str, model_params: dict | None = None, pred_method: str = "direct"):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.pred_method = pred_method
        self.factory = ModelFactory()

    def forecast(
        self,
        history: pd.Series | pd.DataFrame,
        horizon: int,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        model = self.factory.create_model(self.model_name, self.model_params)
        return run_inference(
            model=model,
            history=history,
            horizon=horizon,
            pred_method=self.pred_method,
            X_hist=X_hist,
            X_future=X_future,
            model_builder=lambda: self.factory.create_model(self.model_name, self.model_params),
            model_name=self.model_name,
        )

    def forecast_with_intervals(
        self,
        history: pd.Series | pd.DataFrame,
        horizon: int,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Return DataFrame with columns ['yhat', 'yhat_lower', 'yhat_upper'].

        For 'direct' and 'one_step' pred_methods, delegates to model.predict_with_intervals().
        For 'recursive', intervals are not computable step-by-step, so yhat_lower/upper are NaN.
        """
        method = self.pred_method.lower()
        model = self.factory.create_model(self.model_name, self.model_params)

        if method == "recursive":
            yhat = run_inference(
                model=model,
                history=history,
                horizon=horizon,
                pred_method="recursive",
                X_hist=X_hist,
                X_future=X_future,
                model_builder=lambda: self.factory.create_model(self.model_name, self.model_params),
                model_name=self.model_name,
            )
            return pd.DataFrame({
                "yhat": yhat.values,
                "yhat_lower": np.full(len(yhat), np.nan),
                "yhat_upper": np.full(len(yhat), np.nan),
            })

        if method == "one_step":
            model.fit(history, X_hist=X_hist, X_future=X_future)
            result = model.predict_with_intervals(1, X_future=X_future.iloc[:1].reset_index(drop=True) if X_future is not None else None, alpha=alpha)
        else:  # direct
            model.fit(history, X_hist=X_hist, X_future=X_future)
            result = model.predict_with_intervals(horizon, X_future=X_future, alpha=alpha)

        yhat = _validate_forecast(pd.Series(result["yhat"].values, name="yhat"), len(result), self.model_name)
        result = result.copy()
        result["yhat"] = yhat.values
        return result.reset_index(drop=True)
