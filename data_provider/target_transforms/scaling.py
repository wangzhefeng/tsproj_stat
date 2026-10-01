"""目标缩放：保留拟合参数，供训练还原与预测还原共同使用。"""
from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TargetScaler:
    def __init__(self, method: str = "standard"):
        if method not in {"standard", "minmax"}:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        self.scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    def fit_transform(self, values: pd.Series) -> pd.Series:
        result = self.scaler.fit_transform(values.to_numpy(dtype=float).reshape(-1, 1))
        return pd.Series(result[:, 0], name=values.name)

    def inverse_transform(self, values: pd.Series) -> pd.Series:
        result = self.scaler.inverse_transform(values.to_numpy(dtype=float).reshape(-1, 1))
        return pd.Series(result[:, 0], name=values.name)
