from __future__ import annotations

import pandas as pd


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
