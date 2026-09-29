"""ARIMA 家族模型。

本文件收口 ar/ma/arma/arima/sarima/auto_arima 的主线实现，
阶数搜索 helper 与 statsmodels warning 处理也集中在这里。
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any, Iterable, Protocol, TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults


class _StatsForecastPredict(Protocol):
    # StatsForecast 2.0.1 文档/实现支持 float level，但签名错误地写为 List[int]。
    def __call__(self, h: int, X: np.ndarray | None = None,
                 level: list[float] | None = None) -> dict[str, np.ndarray]: ...

import pandas as pd

from data_provider.data_transfer import to_univariate_series
from models.base import BaseStatModel
from models.exogenous import ExogenousMixin
from .fallbacks import (
    FallbackMixin,
    NaiveModel,
    TrendFallbackModel,
    validate_horizon,
    warn_and_use_fallback,
)


def _normalize_order(order: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if len(order) != 3:
        raise ValueError("order must contain exactly three integers: (p, d, q)")
    normalized = (int(order[0]), int(order[1]), int(order[2]))
    if any(value < 0 for value in normalized):
        raise ValueError("order values must be non-negative")
    return normalized


def _normalize_seasonal_order(order: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    if len(order) != 4:
        raise ValueError("seasonal_order must contain exactly four integers: (P, D, Q, m)")
    normalized = (int(order[0]), int(order[1]), int(order[2]), int(order[3]))
    if any(value < 0 for value in normalized[:3]):
        raise ValueError("seasonal_order values P, D, Q must be non-negative")
    if normalized[3] <= 1:
        raise ValueError("seasonal_order period m must be > 1")
    return normalized


def build_order_grid(
    p_values: Iterable[int] = (0, 1, 2),
    d_values: Iterable[int] = (0, 1),
    q_values: Iterable[int] = (0, 1, 2),
) -> list[tuple[int, int, int]]:
    return [(int(p), int(d), int(q)) for p, d, q in product(p_values, d_values, q_values)]


def select_arima_order(y: pd.Series, order_grid: Iterable[tuple[int, int, int]], ic: str = "aic", exog=None):
    if ic not in {"aic", "bic"}:
        raise ValueError("ic must be one of {'aic', 'bic'}")

    from statsmodels.tsa.arima.model import ARIMA

    best_order = None
    best_score = float("inf")

    for order in order_grid:
        normalized_order = _normalize_order(order)
        try:
            with _fit_warning_context():
                result = ARIMA(y.astype(float), order=normalized_order, **({"exog": exog} if exog is not None else {})).fit()
            score = float(getattr(result, ic))
            if score < best_score:
                best_score = score
                best_order = normalized_order
        except Exception:
            continue

    if best_order is None:
        raise RuntimeError("No valid ARIMA order found in order_grid")

    return best_order, best_score


class _fit_warning_context:
    def __enter__(self):
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.filterwarnings(
            "ignore",
            message=".*Non-invertible starting MA parameters found.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Non-stationary starting autoregressive parameters found.*",
            category=UserWarning,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


class FixedParameterUpdateMixin(ExogenousMixin):
    def update(self, y, X_hist=None):
        """在当前真实历史窗口重新滤波，不重新估计参数。"""
        if self._result is None:
            raise RuntimeError("cannot update an unfitted or fallback model")
        previous_columns = list(getattr(self, "_exog_columns", []))
        exog = self._fit_exog(y, X_hist)
        if previous_columns != self._exog_columns:
            raise ValueError("exogenous schema changed during fixed-parameter update")
        self._result = self._result.apply(to_univariate_series(y).astype(float), exog=exog, refit=False)
        return self


class ARIMAModel(FixedParameterUpdateMixin, FallbackMixin, BaseStatModel):
    def __init__(
        self,
        order: tuple[int, int, int] | list[int] = (1, 1, 1),
        auto_order: bool = False,
        order_grid: Iterable[tuple[int, int, int]] | None = None,
        ic: str = "aic",
    ):
        self.order = _normalize_order(order)
        self.auto_order = auto_order
        self.order_grid = list(order_grid) if order_grid is not None else build_order_grid()
        self.ic = ic
        self.selected_order = self.order
        self.selected_score = None
        self._fallback = NaiveModel()
        self._result = None
        self._train_y = None

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "ARIMAModel":
        series = to_univariate_series(y).astype(float)
        self._train_y = series
        exog = self._fit_exog(y, X_hist, X_future)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.arima.model import ARIMA

            fit_order = self.order
            if self.auto_order:
                fit_order, best_score = select_arima_order(series, self.order_grid, self.ic, exog=exog)
                self.selected_order = fit_order
                self.selected_score = best_score
            else:
                self.selected_order = self.order
                self.selected_score = None

            with _fit_warning_context():
                self._result = ARIMA(series, order=fit_order, **({"exog": exog} if exog is not None else {})).fit()
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name=type(self).__name__,
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        exog = self._predict_exog(horizon, X_future)
        if self._result is None:
            return self._fallback_predict(horizon)
        forecast = self._result.forecast(steps=horizon, exog=exog)
        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(forecast)
        return forecast.reset_index(drop=True).rename("yhat")

    def fitted_values(self) -> pd.Series:
        if self._result is None or self._train_y is None:
            raise ValueError(
                f"{type(self).__name__} has no fitted result; fitted values unavailable"
            )
        return pd.Series(self._result.fittedvalues, name="fitted").reset_index(drop=True)

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        import pandas as pd, numpy as np
        self._predict_exog(horizon, X_future)
        if self._result is None:
            return super().predict_with_intervals(horizon, X_future, alpha)
        try:
            fc = self._result.get_forecast(steps=horizon, exog=self._predict_exog(horizon, X_future))
            mean = fc.predicted_mean.values
            ci = fc.conf_int(alpha=alpha)
            return pd.DataFrame({
                "yhat": mean,
                "yhat_lower": ci.iloc[:, 0].values,
                "yhat_upper": ci.iloc[:, 1].values,
            })
        except Exception:
            return super().predict_with_intervals(horizon, X_future, alpha)


class ARModel(ARIMAModel):
    def __init__(self, p: int = 1):
        if p < 0:
            raise ValueError("p must be non-negative")
        super().__init__(order=(p, 0, 0))
        self.p = p


class MAModel(ARIMAModel):
    def __init__(self, q: int = 1):
        if q < 0:
            raise ValueError("q must be non-negative")
        super().__init__(order=(0, 0, q))
        self.q = q


class ARMAModel(ARIMAModel):
    def __init__(self, p: int = 1, q: int = 1):
        if p < 0 or q < 0:
            raise ValueError("p and q must be non-negative")
        super().__init__(order=(p, 0, q))
        self.p = p
        self.q = q


class SARIMAModel(FixedParameterUpdateMixin, FallbackMixin, BaseStatModel):
    def __init__(
        self,
        order: tuple[int, int, int] | list[int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] | list[int] = (1, 1, 1, 7),
        trend: str | None = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        simple_differencing: bool = False,
        fit_kwargs: dict | None = None,
    ):
        self.order = _normalize_order(order)
        self.seasonal_order = _normalize_seasonal_order(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.simple_differencing = simple_differencing
        self.fit_kwargs: dict[str, Any] = {"disp": False}
        if fit_kwargs is not None:
            self.fit_kwargs.update(fit_kwargs)
        self._fallback = TrendFallbackModel()
        self._result = None
        self._train_y = None

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "SARIMAModel":
        series = to_univariate_series(y).astype(float)
        self._train_y = series
        exog = self._fit_exog(y, X_hist, X_future)
        self._fallback.fit(series)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            with _fit_warning_context():
                result = SARIMAX(
                    series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend=self.trend,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility,
                    simple_differencing=self.simple_differencing,
                    **({"exog": exog} if exog is not None else {}),
                ).fit(**self.fit_kwargs)
                # 默认 fit 返回包装器，公开结果方法由其委托给 SARIMAXResults。
                self._result = cast("SARIMAXResults", result)
        except Exception as exc:
            self._result = None
            warn_and_use_fallback(
                model_name="SARIMAModel",
                fallback_name=type(self._fallback).__name__,
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        exog = self._predict_exog(horizon, X_future)
        if self._result is None:
            return self._fallback_predict(horizon)
        return pd.Series(self._result.forecast(steps=horizon, exog=exog), name="yhat").reset_index(drop=True)

    def fitted_values(self) -> pd.Series:
        if self._result is None or self._train_y is None:
            raise ValueError(
                f"{type(self).__name__} has no fitted result; fitted values unavailable"
            )
        return pd.Series(self._result.fittedvalues, name="fitted").reset_index(drop=True)

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        import pandas as pd, numpy as np
        self._predict_exog(horizon, X_future)
        if self._result is None:
            return super().predict_with_intervals(horizon, X_future, alpha)
        try:
            fc = self._result.get_forecast(steps=horizon, exog=self._predict_exog(horizon, X_future))
            mean = fc.predicted_mean.values
            ci = fc.conf_int(alpha=alpha)
            return pd.DataFrame({
                "yhat": mean,
                "yhat_lower": ci.iloc[:, 0].values,
                "yhat_upper": ci.iloc[:, 1].values,
            })
        except Exception:
            return super().predict_with_intervals(horizon, X_future, alpha)


class AutoARIMAModel(ExogenousMixin, BaseStatModel):
    def __init__(
        self,
        seasonal: bool = False,
        m: int = 1,
        stepwise: bool = True,
        start_p: int = 2,
        start_q: int = 2,
        max_p: int = 5,
        max_q: int = 5,
        max_order: int = 5,
        d: int | None = None,
        test: str = "kpss",
        maxiter: int = 50,
        information_criterion: str = "aic",
        trace: bool = False,
        error_action: str = "ignore",
        suppress_warnings: bool = True,
    ):
        self.seasonal = seasonal
        self.m = m
        self.stepwise = stepwise
        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.max_order = max_order
        self.d = d
        self.test = test
        self.maxiter = maxiter
        self.information_criterion = information_criterion
        self.trace = trace
        self.error_action = error_action
        self.suppress_warnings = suppress_warnings
        self._result = None
        self._train_y: pd.Series | None = None
        self._fallback: ARIMAModel | None = None

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "AutoARIMAModel":
        series = to_univariate_series(y).astype(float)
        self._train_y = series
        exog = self._fit_exog(y, X_hist, X_future)
        self._result = None
        try:
            import pmdarima as pm

            self._result = pm.auto_arima(
                series,
                **({"X": exog} if exog is not None else {}),
                seasonal=self.seasonal,
                m=self.m,
                stepwise=self.stepwise,
                start_p=self.start_p,
                start_q=self.start_q,
                max_p=self.max_p,
                max_q=self.max_q,
                max_order=self.max_order,
                d=self.d,
                test=self.test,
                maxiter=self.maxiter,
                information_criterion=self.information_criterion,
                trace=self.trace,
                error_action=self.error_action,
                suppress_warnings=self.suppress_warnings,
            )
        except Exception as exc:
            self._result = None
            self._ensure_fallback_fitted(series, X_hist=X_hist)
            warn_and_use_fallback(
                model_name="AutoARIMAModel",
                fallback_name=type(self._fallback).__name__ if self._fallback is not None else "ARIMAModel",
                exc=exc,
            )
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        validate_horizon(horizon)
        exog = self._predict_exog(horizon, X_future)
        if self._result is None:
            if self._fallback is None:
                raise RuntimeError("Model is not fitted")
            return self._fallback.predict(horizon, X_future=X_future)
        return pd.Series(self._result.predict(n_periods=horizon, **({"X": exog} if exog is not None else {})), name="yhat")

    def predict_with_intervals(self, horizon: int, X_future=None, alpha: float = 0.05):
        import pandas as pd
        self._predict_exog(horizon, X_future)
        if self._result is None:
            if self._fallback is not None:
                return self._fallback.predict_with_intervals(horizon, X_future, alpha)
            return super().predict_with_intervals(horizon, X_future, alpha)
        try:
            preds, conf_int = self._result.predict(n_periods=horizon, return_conf_int=True, alpha=alpha, X=self._predict_exog(horizon, X_future))
            return pd.DataFrame({
                "yhat": preds,
                "yhat_lower": conf_int[:, 0],
                "yhat_upper": conf_int[:, 1],
            })
        except Exception:
            return super().predict_with_intervals(horizon, X_future, alpha)

    def _ensure_fallback_fitted(self, series: pd.Series, X_hist=None) -> None:
        if self._fallback is None:
            self._fallback = ARIMAModel(auto_order=True)
        self._fallback.fit(series, X_hist=X_hist)


class StatsForecastAutoARIMAModel(ExogenousMixin, BaseStatModel):
    """显式 StatsForecast 后端，不改变 auto_arima 的 pmdarima 默认语义。"""
    def __init__(self, season_length=1, seasonal=False, d=None, D=None,
                 max_p=5, max_q=5, max_P=2, max_Q=2, max_order=5,
                 stepwise=True, ic="aic", approximation=False):
        self.params: dict[str, Any] = dict(season_length=season_length, seasonal=seasonal, d=d, D=D,
                           max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                           max_order=max_order, stepwise=stepwise, ic=ic,
                           approximation=approximation, start_p=min(2, max_p), start_q=min(2, max_q))
        self._result = None
        self._train_y: pd.Series | None = None

    def fit(self, y, X_hist=None, X_future=None):
        from statsforecast.models import AutoARIMA
        exog = self._fit_exog(y, X_hist, X_future)
        self._train_y = to_univariate_series(y).astype(float)
        self._result = AutoARIMA(**self.params).fit(self._train_y.to_numpy(dtype=float), X=exog)
        return self

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
            raise RuntimeError("Model is not fitted")
        pred = self._result.predict(horizon, X=self._predict_exog(horizon, X_future))
        return pd.Series(pred["mean"], name="yhat")

    def predict_with_intervals(self, horizon, X_future=None, alpha=0.05):
        validate_horizon(horizon)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self._result is None:
            raise RuntimeError("Model is not fitted")
        level = round(100 * (1 - alpha), 10)
        predict = cast(_StatsForecastPredict, self._result.predict)
        pred = predict(horizon, X=self._predict_exog(horizon, X_future), level=[level])
        return pd.DataFrame({"yhat": pred["mean"], "yhat_lower": pred[f"lo-{level}"], "yhat_upper": pred[f"hi-{level}"]})

    def predict_with_levels(self, horizon, X_future=None, levels=None, alpha=0.05):
        """SF 后端原生多水平：一次 predict(level=[...]) 返回全部水平列。"""
        from forecasting.intervals import resolve_interval_levels

        validate_horizon(horizon)
        if self._result is None:
            raise RuntimeError("Model is not fitted")
        resolved = resolve_interval_levels(levels, alpha)
        predict = cast(_StatsForecastPredict, self._result.predict)
        sf_levels = [round(level * 100, 10) for level in resolved]
        pred = predict(horizon, X=self._predict_exog(horizon, X_future), level=sf_levels)
        return statsforecast_levels_frame(pred, resolved, len(resolved) > 1)


def statsforecast_levels_frame(pred: dict, levels: list[float], multi: bool) -> pd.DataFrame:
    """StatsForecast 后端 predict(level=[...]) 结果 → 统一多水平列名 DataFrame。

    SF 列键为 lo-{level}/hi-{level}（level 为百分数 float，如 lo-80.0）；
    本项目统一为 yhat_lower[_{label}]/yhat_upper[_{label}]。
    """
    from forecasting.intervals import interval_bound_columns

    data: dict[str, np.ndarray] = {"yhat": pred["mean"]}
    for level in levels:
        sf_level = round(level * 100, 10)
        lower_col, upper_col = interval_bound_columns(level, multi=multi)
        data[lower_col] = pred[f"lo-{sf_level}"]
        data[upper_col] = pred[f"hi-{sf_level}"]
    return pd.DataFrame(data)
