from __future__ import annotations

import warnings

import pandas as pd

from models.base import BaseStatModel


def validate_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")


def to_univariate_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("Input dataframe is empty")
        return y.iloc[:, 0].reset_index(drop=True)
    return y.reset_index(drop=True)


def to_dataframe(y: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        return y.reset_index(drop=True)
    column = y.name or "y"
    return pd.DataFrame({column: y.reset_index(drop=True)})


def warn_and_use_fallback(
    *,
    model_name: str,
    fallback_name: str,
    exc: Exception,
) -> None:
    warnings.warn(f"{model_name} fit failed, fallback to {fallback_name}: {exc}", RuntimeWarning)


class FallbackMixin:
    _fallback: BaseStatModel

    def _fallback_predict(self, horizon: int) -> pd.Series:
        validate_horizon(horizon)
        return self._fallback.predict(horizon)
