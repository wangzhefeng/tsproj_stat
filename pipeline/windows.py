"""原点、历史窗口与未来外生时间对齐；不读文件、不拟合变换器。"""
from __future__ import annotations
import pandas as pd
from data_provider.quality.checks import require_finite
from models.contracts.validation import validate_horizon


def split_history(df: pd.DataFrame, history_size: int) -> pd.DataFrame:
    validate_horizon(history_size)
    if len(df) < history_size:
        raise ValueError("Not enough samples for requested history_size")
    return df.iloc[-history_size:].reset_index(drop=True).copy()


def split_history_future(df: pd.DataFrame, history_size: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_horizon(history_size)
    validate_horizon(horizon)
    if len(df) < history_size + horizon:
        raise ValueError("Not enough samples for requested history_size + horizon")
    return (df.iloc[-(history_size + horizon):-horizon].reset_index(drop=True).copy(),
            df.iloc[-horizon:].reset_index(drop=True).copy())


def align_future_exog(frame: pd.DataFrame, time_col: str, value_cols: list[str],
                      origin: pd.Timestamp, freq: str, horizon: int) -> pd.DataFrame:
    """按原点后的预测时间戳选择已知外生，不按文件前 N 行猜测对齐。"""
    validate_horizon(horizon)
    if frame[time_col].duplicated().any():
        raise ValueError("future exogenous timestamps must be unique")
    expected = future_index(origin, horizon, freq)
    indexed = frame.set_index(time_col)
    if not expected.isin(indexed.index).all():
        raise ValueError("Future exogenous data does not cover requested forecast timestamps")
    result = indexed.loc[expected, value_cols].reset_index(drop=True)
    require_finite(result, "future exogenous data")
    return result


def future_index(origin: pd.Timestamp, horizon: int, freq: str) -> pd.DatetimeIndex:
    """严格晚于原点的频率网格；锚定频率不跳过首个未来点。"""
    validate_horizon(horizon)
    if pd.isna(origin):
        raise ValueError("forecast origin must be a valid timestamp")
    index = pd.date_range(origin, periods=horizon + 1, freq=freq)
    result = index[index > origin][:horizon]
    if len(result) != horizon:
        raise ValueError("frequency must produce increasing future timestamps")
    return result


def forecast_timestamps(history_time: pd.Series | None, horizon: int, freq: str) -> pd.Series:
    """根据有效历史原点生成未来时间；无效输入显式失败。"""
    if history_time is None or history_time.empty:
        raise ValueError("forecast requires a historical timestamp")
    origin = pd.Timestamp(history_time.iloc[-1])
    return pd.Series(future_index(origin, horizon, freq), name="timestamp")
