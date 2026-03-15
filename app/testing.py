from __future__ import annotations

import pandas as pd

from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory


class Tester:
    def __init__(
        self,
        model_name: str,
        model_params: dict | None = None,
        target_col: str = "y",
        initial_train_size: int = 30,
        horizon: int = 7,
        step: int = 7,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.target_col = target_col
        self.initial_train_size = initial_train_size
        self.horizon = horizon
        self.step = step
        self.factory = ModelFactory()

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        model = self.factory.create_model(self.model_name, self.model_params)
        return rolling_backtest(
            df=df,
            model=model,
            target_col=self.target_col,
            initial_train_size=self.initial_train_size,
            horizon=self.horizon,
            step=self.step,
        )
