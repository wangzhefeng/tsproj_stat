from __future__ import annotations

import pandas as pd

from evaluation.backtest import BacktestResult, rolling_backtest
from models.factory import ModelFactory


class Tester:

    def __init__(
        self,
        model_name: str,
        model_params: dict | None = None,
        target_col: str = "y",
        time_col: str = "ds",
        endog_cols: list[str] | None = None,
        exog_cols: list[str] | None = None,
        future_exog_cols: list[str] | None = None,
        initial_train_size: int = 30,
        horizon: int = 7,
        step: int = 7,
        verbose: bool = False,
        progress_every: int = 10,
        n_jobs: int = 1,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.target_col = target_col
        self.time_col = time_col
        self.endog_cols = endog_cols or [target_col]
        self.exog_cols = exog_cols or []
        self.future_exog_cols = future_exog_cols or []
        self.initial_train_size = initial_train_size
        self.horizon = horizon
        self.step = step
        self.verbose = verbose
        self.progress_every = progress_every
        self.n_jobs = n_jobs
        self.factory = ModelFactory()

    def evaluate(self, df: pd.DataFrame) -> BacktestResult:
        model = self.factory.create_model(self.model_name, self.model_params)

        return rolling_backtest(
            df=df,
            model=model,
            target_col=self.target_col,
            time_col=self.time_col,
            endog_cols=self.endog_cols,
            exog_cols=self.exog_cols,
            future_exog_cols=self.future_exog_cols,
            initial_train_size=self.initial_train_size,
            horizon=self.horizon,
            step=self.step,
            verbose=self.verbose,
            progress_every=self.progress_every,
            n_jobs=self.n_jobs,
        )
