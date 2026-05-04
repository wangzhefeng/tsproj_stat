from __future__ import annotations

import numpy as np
import pandas as pd

from models.factory import ModelFactory
from models.inference import normalize_inference_strategy, run_interval_inference, run_point_inference

import os
from pathlib import Path

LOGGING_LABEL = Path(__file__).name[:-3]
os.environ.setdefault("LOG_NAME", LOGGING_LABEL)
from utils.log_util import logger


def _validate_forecast(yhat: pd.Series, horizon: int, model_name: str) -> pd.Series:
    """校验预测长度和数值有效性。

    NaN 可用前后向填充修复；inf 通常代表模型数值发散，必须直接失败。
    """
    if len(yhat) != horizon:
        raise ValueError(f"[{model_name}] forecast length {len(yhat)} != horizon {horizon}")
    nan_count = int(yhat.isna().sum())
    if nan_count > 0:
        logger.warning(
            f"[{model_name}] {nan_count}/{horizon} NaN in forecast, filling with forward/backward fill"
        )
        yhat = yhat.ffill().bfill()
    if np.isinf(yhat.values).any():
        raise ValueError(f"[{model_name}] inf values detected in forecast output")
    return yhat


class Forecaster:
    """预测阶段封装。

    模型创建交给 ModelFactory，多步预测交给 models.inference；
    本类只负责组装参数并做输出质量校验。
    """

    def __init__(
        self,
        model_name: str,
        model_params: dict | None = None,
        inference_strategy: str | None = None,
        pred_method: str | None = None,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.inference_strategy = normalize_inference_strategy(inference_strategy, pred_method)
        self.factory = ModelFactory()

    def forecast(
        self,
        history: pd.Series | pd.DataFrame,
        horizon: int,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        """执行点预测并返回长度等于 horizon 的 yhat 序列。"""
        yhat = run_point_inference(
            model_builder=lambda: self.factory.create_model(self.model_name, self.model_params),
            history=history,
            horizon=horizon,
            inference_strategy=self.inference_strategy,
            X_hist=X_hist,
            X_future=X_future,
        )
        return _validate_forecast(yhat, horizon, self.model_name)

    def forecast_with_intervals(
        self,
        history: pd.Series | pd.DataFrame,
        horizon: int,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """执行区间预测，返回 yhat/yhat_lower/yhat_upper 三列。"""
        result = run_interval_inference(
            model_builder=lambda: self.factory.create_model(self.model_name, self.model_params),
            history=history,
            horizon=horizon,
            inference_strategy=self.inference_strategy,
            X_hist=X_hist,
            X_future=X_future,
            alpha=alpha,
        )
        yhat = _validate_forecast(pd.Series(result["yhat"].values, name="yhat"), len(result), self.model_name)
        result = result.copy()
        result["yhat"] = yhat.values
        return result.reset_index(drop=True)
