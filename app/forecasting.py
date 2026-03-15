from __future__ import annotations

import copy

import pandas as pd

from models.factory import ModelFactory


def run_inference(
    model,
    history: pd.Series | pd.DataFrame,
    horizon: int,
    pred_method: str = "direct",
    model_builder=None,
) -> pd.Series:
    method = pred_method.lower()
    if method not in {"one_step", "recursive", "direct"}:
        raise ValueError("pred_method must be one of {'one_step','recursive','direct'}")

    if method == "one_step":
        model.fit(history)
        return model.predict(1)

    if method == "direct":
        model.fit(history)
        return model.predict(horizon)

    hist = _to_series(history)
    preds = []
    for _ in range(horizon):
        model_i = model_builder() if model_builder is not None else copy.deepcopy(model)
        model_i.fit(hist)
        next_val = float(model_i.predict(1).iloc[0])
        preds.append(next_val)
        hist = pd.concat([hist, pd.Series([next_val])], ignore_index=True)
    return pd.Series(preds, name="yhat")


def _to_series(history: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(history, pd.DataFrame):
        return history.iloc[:, 0].reset_index(drop=True)
    return history.reset_index(drop=True)


class Forecaster:
    def __init__(self, model_name: str, model_params: dict | None = None, pred_method: str = "direct"):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.pred_method = pred_method
        self.factory = ModelFactory()

    def forecast(self, history: pd.Series | pd.DataFrame, horizon: int) -> pd.Series:
        model = self.factory.create_model(self.model_name, self.model_params)
        return run_inference(
            model=model,
            history=history,
            horizon=horizon,
            pred_method=self.pred_method,
            model_builder=lambda: self.factory.create_model(self.model_name, self.model_params),
        )
