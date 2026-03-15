from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStatModel(ABC):
    @abstractmethod
    def fit(self, y: pd.Series | pd.DataFrame) -> "BaseStatModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        raise NotImplementedError

    def forecast(self, horizon: int) -> pd.Series:
        return self.predict(horizon)
