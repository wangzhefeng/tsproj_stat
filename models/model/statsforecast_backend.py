"""StatsForecast 后端适配层：公共基类、列名桥接与全部 SF 系模型。

SF 适配三件套（_StatsForecastModelBase 模板基类、statsforecast_levels_frame
列名桥接、StatsForecastAutoARIMAModel）与 SF 基线候选（auto_ets/auto_theta/
dynamic_theta/auto_ces/random_walk_drift/seasonal_window_average）集中在此，
不再散落于 arima_family 与 baseline_models。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from models.base import BaseStatModel
from models.contracts.inputs import preserve_univariate_series
from models.contracts.intervals import interval_bound_columns, resolve_interval_levels
from models.contracts.validation import validate_horizon
from models.contracts.exogenous import ExogenousMixin
from .arima_family import ARIMAModel
from .fallbacks import FallbackMixin, warn_and_use_fallback


class _StatsForecastPredict(Protocol):
    # StatsForecast 2.0.1 文档/实现支持 float level，但签名错误地写为 List[int]。
    def __call__(self, h: int, X: np.ndarray | None = None,
                 level: list[float] | None = None) -> dict[str, np.ndarray]: ...


def statsforecast_levels_frame(pred: dict, levels: list[float], multi: bool) -> pd.DataFrame:
    """StatsForecast 后端 predict(level=[...]) 结果 → 统一多水平列名 DataFrame。

    SF 列键为 lo-{level}/hi-{level}（level 为百分数 float，如 lo-80.0）；
    本项目统一为 yhat_lower[_{label}]/yhat_upper[_{label}]。
    """
    data: dict[str, np.ndarray] = {"yhat": pred["mean"]}
    for level in levels:
        sf_level = round(level * 100, 10)
        lower_col, upper_col = interval_bound_columns(level, multi=multi)
        data[lower_col] = pred[f"lo-{sf_level}"]
        data[upper_col] = pred[f"hi-{sf_level}"]
    return pd.DataFrame(data)


class _StatsForecastModelBase(BaseStatModel, ABC):
    """StatsForecast 后端模型的公共基类：单序列数组接口、fitted 值与区间协议。

    子类只需声明后端装配；pandas 3 CoW 下统一交付自有可写数组。
    """
    _runtime_backend_fields = ("_sf",)
    def __init__(self, season_length: int = 1, freq: str | None = None):
        if season_length <= 0:
            raise ValueError("season_length must be > 0")
        self.season_length = season_length
        self.freq = freq
        self._sf = None
        self._train_y: pd.Series | None = None
        self._column_name: str | None = None

    @staticmethod
    def _import_statsforecast():
        from statsforecast import StatsForecast

        return StatsForecast

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def fit(self, y: pd.Series | pd.DataFrame, X_hist: pd.DataFrame | None = None, X_future: pd.DataFrame | None = None) -> "_StatsForecastModelBase":
        series = preserve_univariate_series(y)
        # 单序列使用数组接口；freq 保留兼容，不再制造虚拟时间轴。
        # pandas 3 CoW 暴露只读视图，CES 内核会原地工作，必须交付自有可写数组。
        self._train_y = series.astype(float).reset_index(drop=True)
        self._sf = self._build_model().fit(series.to_numpy(dtype=float, copy=True))
        return self

    def fitted_values(self) -> pd.Series:
        # SF 2.0.1 forecast(fitted=True) 需重传训练序列。
        if self._sf is None or self._train_y is None:
            raise ValueError(
                f"{type(self).__name__} has no fitted result; fitted values unavailable"
            )
        fc = self._sf.forecast(self._train_y.to_numpy(dtype=float), 1, fitted=True)
        return pd.Series(np.asarray(fc["fitted"], dtype=float), name="fitted").reset_index(drop=True)

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        if self._sf is None:
            raise RuntimeError("Model is not fitted")
        pred = self._sf.predict(horizon)
        return pd.Series(pred["mean"], dtype=float, name="yhat")

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        validate_horizon(horizon)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self._sf is None:
            raise RuntimeError("Model is not fitted")
        level = round((1 - alpha) * 100, 10)
        pred = self._sf.predict(horizon, level=[level])
        return pd.DataFrame({
            "yhat": pred["mean"],
            "yhat_lower": pred[f"lo-{level}"],
            "yhat_upper": pred[f"hi-{level}"],
        })

    def predict_with_levels(self, horizon: int, X_future=None, levels=None, alpha: float = 0.05):
        """SF 后端原生多水平：一次 predict(level=[...]) 返回全部水平列。"""
        validate_horizon(horizon)
        if self._sf is None:
            raise RuntimeError("Model is not fitted")
        resolved = resolve_interval_levels(levels, alpha)
        sf_levels = [round(level * 100, 10) for level in resolved]
        pred = self._sf.predict(horizon, level=sf_levels)
        return statsforecast_levels_frame(pred, resolved, len(resolved) > 1)


class AutoETSModel(_StatsForecastModelBase):
    """AutoETS（StatsForecast 后端）：自动选择指数平滑误差/趋势/季节形式。"""
    def __init__(
        self,
        season_length: int = 1,
        freq: str | None = None,
        model: str = "ZZZ",
        damped: bool | None = None,
        phi: float | None = None,
    ):
        super().__init__(season_length=season_length, freq=freq)
        self.model = model
        self.damped = damped
        self.phi = phi

    def _build_model(self):
        from statsforecast.models import AutoETS

        return AutoETS(
            season_length=self.season_length,
            model=self.model,
            damped=self.damped,
            phi=self.phi,
        )


class AutoThetaModel(_StatsForecastModelBase):
    """AutoTheta（StatsForecast 后端）：自动 Theta 变体选择。"""
    def __init__(
        self,
        season_length: int = 1,
        freq: str | None = None,
        decomposition_type: str = "multiplicative",
        model: str | None = None,
    ):
        super().__init__(season_length=season_length, freq=freq)
        self.decomposition_type = decomposition_type
        self.model = model

    def _build_model(self):
        from statsforecast.models import AutoTheta

        return AutoTheta(
            season_length=self.season_length,
            decomposition_type=self.decomposition_type,
            model=self.model,
        )


class DynamicThetaModel(_StatsForecastModelBase):
    """DynamicTheta（StatsForecast 后端）：动态优化 Theta。"""
    def __init__(
        self,
        season_length: int = 1,
        freq: str | None = None,
        decomposition_type: str = "multiplicative",
    ):
        super().__init__(season_length=season_length, freq=freq)
        self.decomposition_type = decomposition_type

    def _build_model(self):
        from statsforecast.models import DynamicTheta

        return DynamicTheta(
            season_length=self.season_length,
            decomposition_type=self.decomposition_type,
        )


class AutoCESModel(_StatsForecastModelBase):
    """AutoCES（StatsForecast 后端）：复杂指数平滑自动选择。"""
    def __init__(self, season_length: int = 1, model: str = "Z"):
        super().__init__(season_length)
        self.model = model

    def _build_model(self):
        from statsforecast.models import AutoCES
        return AutoCES(season_length=self.season_length, model=self.model)


class RandomWalkWithDriftModel(_StatsForecastModelBase):
    """带漂移随机游走（StatsForecast 后端）：预测值 = 末值 + 漂移×步长。"""
    def __init__(self):
        super().__init__()

    def _build_model(self):
        from statsforecast.models import RandomWalkWithDrift
        return RandomWalkWithDrift()


class SeasonalWindowAverageModel(_StatsForecastModelBase):
    """季节窗口均值（StatsForecast 后端）：同季位最近 window_size 期均值。"""
    def __init__(self, season_length: int = 7, window_size: int = 2):
        super().__init__(season_length)
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def _build_model(self):
        from statsforecast.models import SeasonalWindowAverage
        return SeasonalWindowAverage(season_length=self.season_length, window_size=self.window_size)


class StatsForecastAutoARIMAModel(ExogenousMixin, BaseStatModel):
    """显式 StatsForecast 后端，不改变 auto_arima 的 pmdarima 默认语义。

    fit 失败回退 ARIMAModel(auto_order=True)，与同族 auto_arima 的
    fallback 语义一致（pmdarima 版回退同为 ARIMAModel）。
    """
    _runtime_backend_fields = ("_result",)
    def __init__(self, season_length=1, seasonal=False, d=None, D=None,
                 max_p=5, max_q=5, max_P=2, max_Q=2, max_order=5,
                 stepwise=True, ic="aic", approximation=False):
        self.params: dict[str, Any] = dict(season_length=season_length, seasonal=seasonal, d=d, D=D,
                           max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                           max_order=max_order, stepwise=stepwise, ic=ic,
                           approximation=approximation, start_p=min(2, max_p), start_q=min(2, max_q))
        self._result = None
        self._train_y: pd.Series | None = None
        self._fallback: ARIMAModel | None = None

    def fit(self, y, X_hist=None, X_future=None):
        from statsforecast.models import AutoARIMA
        from models.contracts.inputs import to_univariate_series

        exog = self._fit_exog(y, X_hist, X_future)
        series = to_univariate_series(y).astype(float)
        self._train_y = series
        try:
            self._result = AutoARIMA(**self.params).fit(series.to_numpy(dtype=float), X=exog)
        except Exception as exc:
            self._result = None
            self._ensure_fallback_fitted(series, X_hist=X_hist)
            warn_and_use_fallback(
                model=self, model_name="StatsForecastAutoARIMAModel",
                fallback_name="ARIMAModel",
                exc=exc,
            )
        return self

    def _ensure_fallback_fitted(self, series: pd.Series, X_hist=None) -> None:
        if self._fallback is None:
            self._fallback = ARIMAModel(auto_order=True)
        self._fallback.fit(series, X_hist=X_hist)

    def fitted_values(self) -> pd.Series:
        # SF 2.0.1 forecast(fitted=True) 需重传训练序列。
        if self._result is None or self._train_y is None:
            raise ValueError(
                f"{type(self).__name__} has no fitted result; fitted values unavailable"
            )
        fc = self._result.forecast(self._train_y.to_numpy(dtype=float), 1, fitted=True)
        return pd.Series(np.asarray(fc["fitted"], dtype=float), name="fitted").reset_index(drop=True)

    def predict(self, horizon, X_future=None):
        validate_horizon(horizon)
        if self._result is None:
            if self._fallback is None:
                raise RuntimeError("Model is not fitted")
            return self._fallback.predict(horizon, X_future=X_future)
        pred = self._result.predict(horizon, X=self._predict_exog(horizon, X_future))
        return pd.Series(pred["mean"], name="yhat")

    def predict_with_intervals(self, horizon, X_future=None, alpha=0.05):
        validate_horizon(horizon)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self._result is None:
            if self._fallback is not None:
                return self._fallback.predict_with_intervals(horizon, X_future, alpha)
            return super().predict_with_intervals(horizon, X_future, alpha)
        level = round(100 * (1 - alpha), 10)
        predict = cast(_StatsForecastPredict, self._result.predict)
        pred = predict(horizon, X=self._predict_exog(horizon, X_future), level=[level])
        return pd.DataFrame({"yhat": pred["mean"], "yhat_lower": pred[f"lo-{level}"], "yhat_upper": pred[f"hi-{level}"]})

    def predict_with_levels(self, horizon, X_future=None, levels=None, alpha=0.05):
        """SF 后端原生多水平：一次 predict(level=[...]) 返回全部水平列。"""
        validate_horizon(horizon)
        if self._result is None:
            if self._fallback is not None:
                return self._fallback.predict_with_levels(horizon, X_future, levels, alpha)
            return super().predict_with_levels(horizon, X_future, levels, alpha)
        resolved = resolve_interval_levels(levels, alpha)
        predict = cast(_StatsForecastPredict, self._result.predict)
        sf_levels = [round(level * 100, 10) for level in resolved]
        pred = predict(horizon, X=self._predict_exog(horizon, X_future), level=sf_levels)
        return statsforecast_levels_frame(pred, resolved, len(resolved) > 1)
