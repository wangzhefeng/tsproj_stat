"""模型基类：BaseStatModel 统一契约（fit/predict/predict_one/区间/拟合值）。

主线模型只实现 fit(y, X_hist, X_future) 与 predict(horizon, X_future)；
单步桥接、多水平区间委托与拟合值门禁的默认实现都在本模块。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from models.registry import ModelSpec

import numpy as np
import pandas as pd


class BaseStatModel(ABC):
    """统计模型统一抽象。

    所有主线模型都通过 fit(y, X_hist=None, X_future=None) 接入训练，
    原生多步 predict(horizon) 是正式接口，predict_one 桥接单步；
    forecasting.strategies 区分 native 执行和旧逐步重拟合策略。
    """

    _model_spec: ModelSpec | None = None
    _ignore_unsupported_inputs = False
    _is_fallback: bool = False
    _fallback_reason: str | None = None

    @abstractmethod
    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X_hist: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
    ) -> "BaseStatModel":
        """拟合模型。y 为目标序列（建模尺度）；X_hist 为历史协变量（含目标列的
        合并帧），X_future 为预测期已知外生。返回 self 以支持链式调用。"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        """原生多步预测：返回长度恰为 horizon 的 yhat 序列。"""
        raise NotImplementedError

    def predict_one(self, X_future_one: pd.DataFrame | None = None) -> float | pd.Series:
        """单步预测桥接方法。

        模型只需实现 predict(horizon)，这里统一抽取 predict(1) 的结果，
        让 recursive/dirrec 策略可以依赖 predict_one 契约。
        """
        pred = self.predict(1, X_future=X_future_one)
        if isinstance(pred, pd.Series):
            if pred.empty:
                raise ValueError("predict(1) returned empty Series")
            return float(pred.iloc[0])
        arr = np.asarray(pred, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("predict(1) returned empty output")
        return float(arr[0])

    def predict_with_intervals(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        返回包含 ['yhat', 'yhat_lower', 'yhat_upper'] 的 DataFrame。
        默认实现：调用 predict() 作为点预测，区间列填 NaN。
        支持区间的子类应重写此方法。
        """
        yhat = self.predict(horizon, X_future)
        return pd.DataFrame(
            {
                "yhat": yhat.values,
                "yhat_lower": np.full(len(yhat), np.nan),
                "yhat_upper": np.full(len(yhat), np.nan),
            }
        )

    def predict_with_levels(
        self,
        horizon: int,
        X_future: pd.DataFrame | None = None,
        levels: Sequence[float] | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """多置信水平区间预测。

        levels 为小数置信水平（如 [0.8, 0.95]）；单水平输出 legacy 列名
        yhat_lower/yhat_upper，多水平输出 yhat_lower_80/yhat_upper_95 式带后缀列。
        默认实现逐水平委托 predict_with_intervals——模型只需实现单水平契约
        即可获得多水平能力（一次 fit 后多次预测，不重复拟合）。
        后端原生支持多水平的模型应重写本方法（如 StatsForecast 后端一次
        predict(level=[...]) 返回全部水平）。
        """
        from forecasting.intervals import interval_bound_columns, resolve_interval_levels

        resolved = resolve_interval_levels(levels, alpha)
        multi = len(resolved) > 1
        frames: dict[str, np.ndarray] = {}
        for level in resolved:
            result = self.predict_with_intervals(horizon, X_future, alpha=1.0 - level)
            lower_col, upper_col = interval_bound_columns(level, multi=multi)
            frames[lower_col] = result["yhat_lower"].to_numpy(dtype=float)
            frames[upper_col] = result["yhat_upper"].to_numpy(dtype=float)
            if "yhat" not in frames:
                frames["yhat"] = result["yhat"].to_numpy(dtype=float)
        ordered = {"yhat": frames.pop("yhat")}
        ordered.update(frames)
        return pd.DataFrame(ordered)

    def fitted_values(self) -> pd.Series:
        """返回训练期一步-ahead 拟合值（in-sample fitted values）。

        默认 RAISE：只有后端提供拟合值且 registry 声明 supports_fitted_values
        的模型才提供本能力；不伪造、不降级（naive 类基线的一步拟合值
        是 t-1 平移，语义争议大，不纳入主线）。
        """
        raise ValueError(
            f"{type(self).__name__} does not provide fitted values "
            "(backend support required; see registry supports_fitted_values)"
        )
