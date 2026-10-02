"""波动率模型。

ARCH/GARCH 作为 optional 模型接入统一 fit/predict 契约；
依赖不可用或拟合失败时回退到简单预测，避免主流程异常中断。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from models.contracts.inputs import to_univariate_series
from models.contracts.validation import validate_horizon
from .fallbacks import (
    NaiveModel,
    FallbackMixin,
    warn_and_use_fallback,
)


class _ARCHFamilyModel(FallbackMixin, BaseStatModel):
    """arch 库后端模板：子类只声明 vol/p/q，方法全部复用。"""
    _runtime_backend_fields = ("_result",)
    _vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH"
    _p: int = 1
    _q: int = 1

    def __init__(self):
        self._result = None
        self._fallback = NaiveModel()

    def _build_backend(self, series):
        from arch import arch_model

        return arch_model(series, mean="Constant", vol=self._vol, p=self._p, q=self._q)

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None):
        series = to_univariate_series(y).astype(float)
        self._fallback.fit(series)
        try:
            self._result = self._build_backend(series).fit(disp="off")
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model=self, model_name=type(self).__name__,
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._result is None:
            return self._fallback_predict(horizon)
        fc = self._result.forecast(horizon=horizon)
        values = np.asarray(fc.mean.iloc[-1]).reshape(-1)
        return pd.Series(values[:horizon], name="yhat")

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        from scipy import stats

        if self._result is None:
            return super().predict_with_intervals(horizon, X_future, alpha)
        try:
            fc = self._result.forecast(horizon=horizon)
            mean = np.asarray(fc.mean.iloc[-1]).reshape(-1)[:horizon]
            var = np.asarray(fc.variance.iloc[-1]).reshape(-1)[:horizon]
            z = stats.norm.ppf(1 - alpha / 2)
            std = np.sqrt(np.maximum(var, 0))
            return pd.DataFrame({
                "yhat": mean,
                "yhat_lower": mean - z * std,
                "yhat_upper": mean + z * std,
            })
        except Exception:
            return super().predict_with_intervals(horizon, X_future, alpha)


class ARCHModel(_ARCHFamilyModel):
    """自回归条件异方差模型（ARCH(1)）。"""
    _vol = "ARCH"
    _p = 1
    _q = 0


class GARCHModel(_ARCHFamilyModel):
    """广义自回归条件异方差模型（GARCH(1,1)）。"""
    _vol = "GARCH"
    _p = 1
    _q = 1
