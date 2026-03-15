from __future__ import annotations

import pandas as pd

from .models.factory import ModelFactory


class Trainer:
    def __init__(self, model_name: str, model_params: dict | None = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.factory = ModelFactory()

    def train(self, y: pd.Series | pd.DataFrame):
        model = self.factory.create_model(self.model_name, self.model_params)
        model.fit(y)
        return model
