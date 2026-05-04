from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStatModel(ABC):
    """统计模型统一抽象。

    所有主线模型都通过 fit(y, X_hist=None, X_future=None) 接入训练，
    多步推理由 models.inference 统一编排，模型自身只需保证 predict/predict_one 契约。
    """

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
        """单步预测桥接方法。

        迁移期仍允许模型只实现 predict(1)，这里统一抽取第一步结果，
        让 recursive/dirrec 策略可以依赖 predict_one 契约。
        """
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
