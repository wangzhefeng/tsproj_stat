from __future__ import annotations

import pandas as pd


def run_inference(model, history: pd.Series | pd.DataFrame, horizon: int) -> pd.Series:
    model.fit(history)
    return model.predict(horizon)
