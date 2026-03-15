from __future__ import annotations

import pandas as pd


class Forecaster:
    def __init__(self, model):
        self.model = model

    def forecast(self, horizon: int) -> pd.Series:
        return self.model.predict(horizon)
