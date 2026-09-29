"""预测引擎：多步策略、预测区间与预测器组装。"""
from .strategies import FORECAST_STRATEGIES, WINDOW_MODES

__all__ = ["FORECAST_STRATEGIES", "WINDOW_MODES"]
