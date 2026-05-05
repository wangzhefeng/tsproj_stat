from __future__ import annotations

import pandas as pd


def validate_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")


def to_univariate_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    """将 Series/DataFrame 统一转为单目标序列，默认取 DataFrame 第一列。"""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("Input dataframe is empty")
        return y.iloc[:, 0].reset_index(drop=True)
    return y.reset_index(drop=True)


def to_dataframe(y: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """将单目标序列包装为 DataFrame，便于与 X_hist 合并。"""
    if isinstance(y, pd.DataFrame):
        return y.reset_index(drop=True)
    column = y.name or "y"
    return pd.DataFrame({column: y.reset_index(drop=True)})


def combine_history_frame(
    y: pd.Series | pd.DataFrame,
    X_hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """合并目标序列与历史多源输入，并保证目标列排在第一列。"""
    if X_hist is None:
        return to_dataframe(y).astype(float)

    frame = X_hist.reset_index(drop=True).copy()
    target = to_univariate_series(y).astype(float).reset_index(drop=True)
    target_name = target.name or (frame.columns[0] if len(frame.columns) else "y")

    if len(frame) != len(target):
        raise ValueError("X_hist must have the same number of rows as y")

    if target_name in frame.columns:
        frame[target_name] = target.values
    else:
        frame.insert(0, target_name, target.values)

    ordered = [target_name, *[col for col in frame.columns if col != target_name]]
    return frame[ordered].astype(float)
