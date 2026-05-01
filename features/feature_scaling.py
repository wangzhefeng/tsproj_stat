from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureScaler:
    def __init__(self, scaler_type: str = "standard"):
        if scaler_type not in {"standard", "minmax"}:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.fitted_columns: list[str] = []

    def fit_transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        columns = columns or list(df.columns)
        self.fitted_columns = columns
        out = df.copy()
        out[columns] = self.scaler.fit_transform(out[columns])
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_columns:
            raise RuntimeError("FeatureScaler is not fitted")
        out = df.copy()
        out[self.fitted_columns] = self.scaler.transform(out[self.fitted_columns])
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_columns:
            raise RuntimeError("FeatureScaler is not fitted")
        out = df.copy()
        out[self.fitted_columns] = self.scaler.inverse_transform(out[self.fitted_columns])
        return out
