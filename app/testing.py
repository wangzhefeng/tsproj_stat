from __future__ import annotations

import pandas as pd

from evaluation.backtest import BacktestResult, rolling_backtest
from models.factory import ModelFactory
from models.inference import normalize_inference_strategy, normalize_window_mode


class Tester:
    """回测阶段封装。

    将 AppConfig 中的数据列、窗口参数和推理策略传给 rolling_backtest，
    具体窗口切分与指标计算仍集中在 evaluation/backtest.py。
    """

    def __init__(
        self,
        model_name: str,
        model_params: dict | None = None,
        target_col: str = "y",
        time_col: str = "ds",
        endog_cols: list[str] | None = None,
        exog_cols: list[str] | None = None,
        future_exog_cols: list[str] | None = None,
        train_size: int = 30,
        horizon: int = 7,
        step: int = 7,
        inference_strategy: str = "direct",
        window_mode: str = "expanding",
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
        self.train_size = train_size
        self.horizon = horizon
        self.step = step
        self.inference_strategy = normalize_inference_strategy(inference_strategy, None)
        self.window_mode = normalize_window_mode(window_mode)
        self.verbose = verbose
        self.progress_every = progress_every
        self.n_jobs = n_jobs
        self.factory = ModelFactory()

    def evaluate(self, df: pd.DataFrame) -> BacktestResult:
        """执行 rolling backtest 并返回结构化结果。"""
        return rolling_backtest(
            df=df,
            model_builder=lambda: self.factory.create_model(self.model_name, self.model_params),
            target_col=self.target_col,
            time_col=self.time_col,
            endog_cols=self.endog_cols,
            exog_cols=self.exog_cols,
            future_exog_cols=self.future_exog_cols,
            train_size=self.train_size,
            horizon=self.horizon,
            step=self.step,
            inference_strategy=self.inference_strategy,
            window_mode=self.window_mode,
            verbose=self.verbose,
            progress_every=self.progress_every,
            n_jobs=self.n_jobs,
        )
