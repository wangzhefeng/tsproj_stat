from __future__ import annotations

import copy

import pandas as pd

from models.factory import ModelFactory
from data_provider.data_transfer import combine_history_frame, to_univariate_series


def run_inference(
    model,
    history: pd.Series | pd.DataFrame,
    horizon: int,
    pred_method: str = "direct",
    X_hist: pd.DataFrame | None = None,
    X_future: pd.DataFrame | None = None,
    model_builder=None,
) -> pd.Series:
    # model forecasting method
    method = pred_method.lower()
    if method not in {"one_step", "recursive", "direct"}:
        raise ValueError("pred_method must be one of {'one_step','recursive','direct'}")
    # one step forecasting
    if method == "one_step":
        model.fit(history, X_hist=X_hist, X_future=X_future)
        return model.predict(1, X_future=X_future.iloc[:1].reset_index(drop=True) if X_future is not None else None)
    # direct forecasting
    if method == "direct":
        model.fit(history, X_hist=X_hist, X_future=X_future)
        return model.predict(horizon, X_future=X_future)
    # recursive forecasting
    hist = to_univariate_series(history)
    hist_frame = combine_history_frame(history, X_hist)
    preds = []
    for _ in range(horizon):
        model_i = model_builder() if model_builder is not None else copy.deepcopy(model)
        next_future = None
        if X_future is not None:
            step_idx = len(preds)
            if step_idx >= len(X_future):
                raise ValueError("Recursive forecasting requires complete future exogenous rows for every forecast step")
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
    
    return pd.Series(preds, name="yhat")


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
        # model building
        model = self.factory.create_model(self.model_name, self.model_params)
        # model inference
        return run_inference(
            model=model,
            history=history,
            horizon=horizon,
            pred_method=self.pred_method,
            X_hist=X_hist,
            X_future=X_future,
            model_builder=lambda: self.factory.create_model(self.model_name, self.model_params),
        )
