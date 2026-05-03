from __future__ import annotations

from abc import ABC, abstractmethod

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

    def forecast(self, horizon: int) -> pd.Series:
        return self.predict(horizon)
