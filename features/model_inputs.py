"""窗口内建模特征与未来派生值；时间轴由调用方提供，不读取未来目标。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from features.feature_engineering import build_history_features


@dataclass(frozen=True)
class ModelFeatureSpec:
    datetime_features: bool = True
    lags: tuple[int, ...] = ()

    def prepare(self, frame: pd.DataFrame, history_time: pd.Series | None,
                future_time: pd.Series | None) -> tuple[pd.DataFrame, "FutureFeatures", int]:
        source = frame.reset_index(drop=True).copy()
        if self.datetime_features:
            if history_time is None or len(history_time) != len(source):
                raise ValueError("model_input datetime features require aligned historical timestamps")
            source["__feature_time__"] = pd.to_datetime(history_time).to_numpy()
        derived, columns, warmup = build_history_features(
            source, "__feature_time__", str(frame.columns[0]), self.datetime_features, list(self.lags))
        if set(columns) & set(frame.columns):
            raise ValueError("derived feature names collide with historical input columns")
        if warmup >= len(frame):
            raise ValueError("feature warmup leaves no training observations")
        context = FutureFeatures(self, frame.iloc[:, 0].astype(float).reset_index(drop=True), future_time)
        combined = pd.concat([frame.reset_index(drop=True), derived], axis=1).iloc[warmup:]
        return combined.astype(float).reset_index(drop=True), context, warmup


@dataclass(frozen=True)
class FutureFeatures:
    spec: ModelFeatureSpec
    history_y: pd.Series
    future_time: pd.Series | None

    def row(self, future: pd.DataFrame | None, step: int, predictions: list[float]) -> pd.DataFrame:
        """第 step 行只依赖窗口历史及已经生成的预测，绝不借验证标签。"""
        if future is not None and step >= len(future):
            raise ValueError("future exogenous rows are insufficient for derived features")
        result = future.iloc[step:step + 1].reset_index(drop=True).copy() if future is not None else pd.DataFrame(index=[0])
        values: dict[str, float] = {}
        if self.spec.datetime_features:
            if self.future_time is None or step >= len(self.future_time):
                raise ValueError("model_input datetime features require aligned future timestamps")
            timestamp = pd.Timestamp(self.future_time.iloc[step])
            values.update(hour=timestamp.hour, dayofweek=timestamp.dayofweek,
                          month=timestamp.month, dayofyear=timestamp.dayofyear)
        history = self.history_y.tolist() + predictions
        for lag in self.spec.lags:
            if len(predictions) != step or lag > len(history):
                raise ValueError("insufficient as-of history for target lag features")
            values[f"lag_{lag}"] = float(history[-lag])
        if set(values) & set(result.columns):
            raise ValueError("future exogenous input must not supply derived feature columns")
        for name, value in values.items():
            result[name] = float(value)
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("non-finite future model inputs")
        return result
