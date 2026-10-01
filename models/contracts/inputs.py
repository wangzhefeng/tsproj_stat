"""建模链路的序列形状工具：单目标归一、历史帧合并与频率解析。

被 forecasting 与 models 层共同消费；不放 forecasting/ 是为了避免
models → forecasting 的反向依赖。本模块只做 pandas 形状归一，不做数据加载/清洗。
"""
from __future__ import annotations

import pandas as pd


def to_univariate_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    """将 Series/DataFrame 统一转为单目标序列，默认取 DataFrame 第一列。"""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("Input dataframe is empty")
        return y.iloc[:, 0].reset_index(drop=True)
    return y.reset_index(drop=True)


def preserve_univariate_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    """取单目标序列并保留原索引（不 reset），未命名序列统一命名为 y。

    与 to_univariate_series 的差别：StatsForecast 等后端需要保留 DatetimeIndex
    做频率推断，因此本函数不重置索引。
    """
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("Input dataframe is empty")
        series = y.iloc[:, 0].copy()
    else:
        series = y.copy()
    series.name = series.name or "y"
    return series.astype(float)


def resolve_series_freq(series: pd.Series, fallback_freq: str | None = None) -> str:
    """解析序列频率：DatetimeIndex 自带或推断优先，缺省回退 fallback_freq 或 "D"。"""
    if isinstance(series.index, pd.DatetimeIndex):
        inferred = series.index.freqstr or pd.infer_freq(series.index)
        if inferred:
            return inferred
    return fallback_freq or "D"


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
    # 未命名 Series 使用统一默认目标名，不能将第一列协变量覆盖成目标。
    target_name = target.name if target.name is not None else "y"

    if len(frame) != len(target):
        raise ValueError("X_hist must have the same number of rows as y")

    if target_name in frame.columns:
        frame[target_name] = target.values
    else:
        frame.insert(0, target_name, target.to_numpy(dtype=float))

    ordered = [target_name, *[col for col in frame.columns if col != target_name]]
    return frame.loc[:, ordered].astype(float)
