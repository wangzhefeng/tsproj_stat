from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStatModel(ABC):

    @abstractmethod
    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "BaseStatModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        raise NotImplementedError

    def predict_one(self, X_future_one: pd.DataFrame | None = None) -> float | pd.Series:
        """Compatibility bridge while the project migrates to a single-step contract."""
        pred = self.predict(1, X_future=X_future_one)
        if isinstance(pred, pd.Series):
            if pred.empty:
                raise ValueError("predict(1) returned empty Series")
            return float(pred.iloc[0])
        arr = np.asarray(pred, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("predict(1) returned empty output")
        return float(arr[0])

    def predict_with_intervals(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        返回包含 ['yhat', 'yhat_lower', 'yhat_upper'] 的 DataFrame。
        默认实现：调用 predict() 作为点预测，区间列填 NaN。
        支持区间的子类应重写此方法。
        """
        yhat = self.predict(horizon, X_future)
        return pd.DataFrame(
            {
                "yhat": yhat.values,
                "yhat_lower": np.full(len(yhat), np.nan),
                "yhat_upper": np.full(len(yhat), np.nan),
            }
        )

    def forecast(self, horizon: int) -> pd.Series:
        return self.predict(horizon)
