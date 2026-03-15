from __future__ import annotations

import pandas as pd


class FeatureEngineer:
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
        lags = lags or []
        out = df.copy()

        if self.time_col in out.columns:
            out[self.time_col] = pd.to_datetime(out[self.time_col])

        if enable_datetime_features and self.time_col in out.columns:
            out["hour"] = out[self.time_col].dt.hour
            out["dayofweek"] = out[self.time_col].dt.dayofweek
            out["month"] = out[self.time_col].dt.month
            out["dayofyear"] = out[self.time_col].dt.dayofyear

        for lag in lags:
            out[f"lag_{lag}"] = out[self.target_col].shift(lag)

        target_shift_cols = []
        for step in range(1, horizon + 1):
            col = f"target_t_plus_{step}"
            out[col] = out[self.target_col].shift(-step)
            target_shift_cols.append(col)

        feature_cols = [c for c in out.columns if c not in {self.time_col, self.target_col} and not c.startswith("target_t_plus_")]
        out = out.dropna().reset_index(drop=True)
        return out, feature_cols, target_shift_cols
