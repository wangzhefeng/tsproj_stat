from __future__ import annotations

import pandas as pd

from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory
from models.inference import normalize_inference_strategy

import os
from pathlib import Path
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ.setdefault('LOG_NAME', LOGGING_LABEL)
from utils.log_util import logger


class AutoSelector:
    """Rank candidate models via a mini rolling backtest and return the best model name.

    Uses a small number of backtest windows (n_windows) to keep evaluation fast.
    The model with the lowest score on `metric` wins.
    """

    def __init__(
        self,
        candidates: list[str],
        metric: str = "mae",
        n_windows: int = 5,
        initial_train_size: int = 30,
        horizon: int = 7,
        model_params_map: dict[str, dict] | None = None,
        inference_strategy: str = "direct",
    ):
        if not candidates:
            raise ValueError("candidates must not be empty")
        if metric not in {"mae", "rmse", "mape", "smape", "mse", "r2", "bias", "max_error"}:
            raise ValueError(f"metric must be a valid backtest metric, got: {metric!r}")
        if n_windows <= 0:
            raise ValueError("n_windows must be > 0")
        self.candidates = candidates
        self.metric = metric
        self.n_windows = n_windows
        self.initial_train_size = initial_train_size
        self.horizon = horizon
        self.model_params_map = model_params_map or {}
        self.inference_strategy = normalize_inference_strategy(inference_strategy, None)
        self._scores: dict[str, float] = {}

    def select(
        self,
        y: pd.Series,
        X_hist: pd.DataFrame | None = None,
        target_col: str = "y",
        time_col: str = "ds",
    ) -> str:
        """Evaluate all candidates and return the name of the best one."""
        n = len(y)
        total_needed = self.initial_train_size + self.horizon
        if n < total_needed:
            raise ValueError(
                f"AutoSelector needs at least {total_needed} data points, got {n}"
            )

        # Compute step so we get at most n_windows windows
        available = n - total_needed
        step = max(1, available // self.n_windows)

        df = y.to_frame(name=target_col) if isinstance(y, pd.Series) else y.copy()
        if X_hist is not None:
            for col in X_hist.columns:
                if col not in df.columns:
                    df[col] = X_hist[col].values

        factory = ModelFactory()
        self._scores = {}

        for model_name in self.candidates:
            params = self.model_params_map.get(model_name, {})
            try:
                result = rolling_backtest(
                    df=df,
                    model_builder=lambda model_name=model_name, params=params: factory.create_model(model_name, params),
                    target_col=target_col,
                    time_col=time_col if time_col in df.columns else None,
                    train_size=self.initial_train_size,
                    horizon=self.horizon,
                    step=step,
                    inference_strategy=self.inference_strategy,
                    verbose=False,
                )
                score = result.summary.get(self.metric, float("inf"))
                self._scores[model_name] = float(score)
                logger.info(
                    f"[AutoSelect] {model_name}: {self.metric}={score:.4f} "
                    f"({result.summary['window_count']} windows)"
                )
            except Exception as exc:
                logger.warning(f"[AutoSelect] {model_name} failed: {exc}")
                self._scores[model_name] = float("inf")

        valid = {k: v for k, v in self._scores.items() if v < float("inf")}
        if not valid:
            raise RuntimeError("AutoSelector: all candidate models failed evaluation")

        best = min(valid, key=valid.__getitem__)
        logger.info(
            f"[AutoSelect] selected: {best!r} ({self.metric}={valid[best]:.4f}) "
            f"from {list(valid.keys())}"
        )
        return best

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._scores)
