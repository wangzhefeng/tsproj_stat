from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseForecaster(ABC):
    """预测模型抽象基类。"""

    @abstractmethod
    def fit(self, y: pd.Series) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        ...
