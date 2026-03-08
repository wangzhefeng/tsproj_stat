from __future__ import annotations

import pandas as pd


def run_inference(model, history: pd.Series, horizon: int) -> pd.Series:
    """统一推理入口：fit + predict。"""
    model.fit(history)
    return model.predict(horizon)
