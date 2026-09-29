"""回归型统计模型的外生输入契约，独立于具体后端。"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ExogenousMixin:
    def _fit_exog(self, y, X_hist=None, X_future=None):
        target = y.columns[0] if isinstance(y, pd.DataFrame) else (y.name or "y")
        frame = None if X_hist is None else X_hist.drop(columns=[target], errors="ignore").copy()
        if frame is not None and len(frame) != len(y):
            raise ValueError("historical exogenous rows must match y")
        self._exog_columns = [] if frame is None else list(frame.columns)
        if len(set(self._exog_columns)) != len(self._exog_columns):
            raise ValueError("duplicate historical exogenous columns")
        if frame is None or not self._exog_columns:
            if X_future is not None and X_future.shape[1]:
                raise ValueError("future exogenous inputs require historical exogenous columns")
            return None
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("historical exogenous inputs must be finite")
        if X_future is not None:
            self._predict_exog(len(X_future), X_future)
        return values

    def _predict_exog(self, horizon, X_future):
        columns = getattr(self, "_exog_columns", [])
        if not columns:
            if X_future is not None and X_future.shape[1]:
                raise ValueError("unexpected future exogenous inputs")
            return None
        if X_future is None or len(X_future) != horizon:
            raise ValueError(f"future exogenous inputs require exactly {horizon} rows")
        if not X_future.columns.is_unique or set(X_future.columns) != set(columns):
            raise ValueError(f"future exogenous columns must match {columns}")
        values = X_future[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("future exogenous inputs must be finite")
        return values
