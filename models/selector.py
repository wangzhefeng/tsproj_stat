from __future__ import annotations

import pandas as pd

from evaluation.backtest import rolling_backtest
from models.factory import ModelFactory
from models.inference import normalize_inference_strategy
from models.registry import MODEL_REGISTRY

import os
from pathlib import Path
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ.setdefault('LOG_NAME', LOGGING_LABEL)
from utils.log_util import logger


class AutoSelector:
    """用小规模 rolling backtest 对候选模型排序，并返回最优模型名。

    通过 n_windows 控制评估成本；除 r2 以外，当前按 metric 越小越优选择模型。
    """

    def __init__(
        self,
        candidates: list[str] | None = None,
        metric: str = "mae",
        n_windows: int = 5,
        initial_train_size: int = 30,
        horizon: int = 7,
        model_params_map: dict[str, dict] | None = None,
        inference_strategy: str = "direct",
    ):
        if candidates is None:
            candidates = [name for name, spec in MODEL_REGISTRY.items() if spec.stability == "stable"]
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
        """评估全部候选模型，并返回有效得分最优的模型名。"""
        n = len(y)
        total_needed = self.initial_train_size + self.horizon
        if n < total_needed:
            raise ValueError(
                f"AutoSelector needs at least {total_needed} data points, got {n}"
            )

        # 计算窗口步长，使自动选择最多评估 n_windows 个窗口，避免 CLI 默认运行过慢。
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

        best = max(valid, key=valid.__getitem__) if self.metric == "r2" else min(valid, key=valid.__getitem__)
        logger.info(
            f"[AutoSelect] selected: {best!r} ({self.metric}={valid[best]:.4f}) "
            f"from {list(valid.keys())}"
        )
        return best

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._scores)
