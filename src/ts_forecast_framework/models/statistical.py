from __future__ import annotations

import warnings

import pandas as pd

from .base import BaseForecaster
from .selection import build_order_grid, select_arima_order


class NaiveForecaster(BaseForecaster):
    """最简单基线模型：未来值等于最后一个观测值。"""

    def __init__(self):
        self.last_value = None

    def fit(self, y: pd.Series):
        if len(y) == 0:
            raise ValueError("Input series is empty")
        self.last_value = float(y.iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self.last_value is None:
            raise RuntimeError("Model is not fitted")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        return pd.Series([self.last_value] * horizon, name="yhat")


class ARIMAForecaster(BaseForecaster):
    """基于 statsmodels 的 ARIMA 预测器，支持 AIC/BIC 自动选阶。"""

    def __init__(self, order=(1, 1, 1), auto_order: bool = False, order_grid=None, ic: str = "aic"):
        self.order = order
        self.auto_order = auto_order
        self.order_grid = list(order_grid) if order_grid is not None else build_order_grid()
        self.ic = ic

        self.selected_order = order
        self.selected_score = None

        self._naive = NaiveForecaster()
        self._result = None

    def fit(self, y: pd.Series):
        if len(y) == 0:
            raise ValueError("Input series is empty")

        self._naive.fit(y)

        try:
            from statsmodels.tsa.arima.model import ARIMA

            fit_order = self.order
            if self.auto_order:
                fit_order, best_score = select_arima_order(y, self.order_grid, self.ic)
                self.selected_order = fit_order
                self.selected_score = best_score
            else:
                self.selected_order = self.order
                self.selected_score = None

            model = ARIMA(y.astype(float), order=fit_order)
            self._result = model.fit()
        except Exception as e:
            self._result = None
            warnings.warn(
                f"ARIMA fit failed, fallback to NaiveForecaster. reason={e}",
                RuntimeWarning,
            )

        return self

    def predict(self, horizon: int) -> pd.Series:
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        if self._result is None:
            return self._naive.predict(horizon)

        forecast = self._result.forecast(steps=horizon)
        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(forecast)
        forecast.name = "yhat"
        forecast = forecast.reset_index(drop=True)
        return forecast
