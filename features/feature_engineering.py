"""派生特征：历史输入特征与分析快照共用计算；未来标签仅供分析。"""
from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """分析型特征构造器。

    create_features 输出分析快照；build_history_features 用于显式模型输入，
    不生成未来标签、不填补 lag warmup。
    """

    def __init__(self, time_col: str = "ds", target_col: str = "y"):
        self.time_col = time_col
        self.target_col = target_col

    def create_features(
        self,
        df: pd.DataFrame,
        enable_datetime_features: bool = True,
        lags: list[int] | None = None,
        horizon: int = 1,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """创建时间特征、滞后特征和未来 target shift 列。

        返回前整表 dropna：同时丢弃 lag warmup 头部行、target shift 尾部行
        及原始列含缺失的行；快照语义为「所有列均完整的样本视图」，
        不适用于含真实缺失的训练数据构造。
        """
        lags = lags or []
        out = df.copy()

        if self.time_col in out.columns:
            out[self.time_col] = pd.to_datetime(out[self.time_col])

        derived, _, _ = build_history_features(out, self.time_col, self.target_col,
                                               enable_datetime_features, lags)
        out = pd.concat([out, derived], axis=1)

        target_shift_cols = []
        for step in range(1, horizon + 1):
            col = f"target_t_plus_{step}"
            out[col] = out[self.target_col].shift(-step)
            target_shift_cols.append(col)

        feature_cols = [c for c in out.columns if c not in {self.time_col, self.target_col} and not c.startswith("target_t_plus_")]
        out = out.dropna().reset_index(drop=True)

        return out, feature_cols, target_shift_cols


def build_history_features(df: pd.DataFrame, time_col: str, target_col: str,
                           enable_datetime_features: bool, lags: list[int]) -> tuple[pd.DataFrame, list[str], int]:
    """只派生历史可用的列，保留缺失行供调用者同步切掉 warmup。

    返回 (特征帧, 特征列名, warmup 深度)；warmup = max(lags)，即头部
    因 lag shift 产生 NaN 的行数，调用方据此同步收缩对齐的历史视图。
    """
    out = pd.DataFrame(index=df.index)
    if enable_datetime_features and time_col in df:
        dt = pd.to_datetime(df[time_col])
        for name, values in {"hour": dt.dt.hour, "dayofweek": dt.dt.dayofweek,
                             "month": dt.dt.month, "dayofyear": dt.dt.dayofyear}.items():
            out[name] = values
    for lag in lags:
        if lag <= 0:
            raise ValueError("feature lags must be positive")
        out[f"lag_{lag}"] = df[target_col].shift(lag)
    warmup = max(lags) if lags else 0
    return out, list(out.columns), warmup
